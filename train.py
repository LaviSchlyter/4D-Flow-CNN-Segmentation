# ==================================================================
# import 
# ==================================================================
import shutil
import logging
import os.path
import numpy as np
import torch
import utils
import model as model
import config.system as sys_config
import data_freiburg_numpy_to_hdf5
import wandb
import losses 
import torch.nn.functional as F
from tqdm import tqdm
import h5py
from pytorch_model_summary import summary
from torch.optim.lr_scheduler import CyclicLR

import model_zoo
# ==================================================================
# Set seed for reproducibility
# ==================================================================

SEED = 0 
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import unet as exp_config

# ==================================================================
# Set up logging
# ==================================================================
log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
print('log_dir: ', log_dir)

# ==================================================================
# Save checkpoints
# ==================================================================

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

# ==================================================================
# Iterate over mini-batches
# ==================================================================

def iterate_minibatches(images, labels, batch_size, cut_t_slices = False, n_cut_slices = 5):
    """
    Function to create mini batches from the dataset of a certain batch size
    :param images: input images
    :param labels: labels
    :param batch_size: batch size
    :return: mini batches"""
    assert len(images) == len(labels)
    
    # Generate randomly selected slices in each minibatch

    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # Use only fraction of the batches in each epoch

    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue


        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]
        if cut_t_slices:
            X = X[:, :, :, n_cut_slices:-n_cut_slices, :]
            y = y[:, :, :, n_cut_slices:-n_cut_slices]
        
        # ===========================
        # check if the velocity fields are to be used for the segmentation...
        # ===========================
        if exp_config.nchannels == 1:
            X = X[..., 1:2]
    
        # ===========================
        # augment the batch            
        # ===========================
        if exp_config.da_ratio > 0.0 and exp_config.nchannels == 1:
            X, y = utils.augment_data(X,
                                      y,
                                      data_aug_ratio = exp_config.da_ratio)
        
        yield X, y



# ==================================================================
# ==================================================================
def evaluate_losses(labels_pred,
                    labels,
                    nlabels,
                    loss_type,
                    save_results = False,
                    label_hot_bool = False):
    """
    A function to compute various loss measures to compare the predicted and ground truth annotations
    """
   
    loss_ = loss_function(labels_pred, labels, nlabels, loss_type, labels_as_1hot = label_hot_bool)

    labels = torch.nn.functional.one_hot(labels.long(), num_classes = nlabels)
    labels = labels.transpose(1,4).transpose(2,4).transpose(3,4)

    _, dice_, _ = losses.compute_dice(labels_pred, labels)

    return loss_, dice_


def loss_function(labels_pred, labels, nlabels, loss_type, labels_as_1hot = False):
    """
    Function to compute the loss
    :param labels_pred: predicted inputs before the softmax
    :param labels: ground truth labels
    :param nlabels: number of GT labels
    :param loss_type: type of loss function
    :return: loss"""

    if labels_as_1hot is False:
        
        labels = torch.nn.functional.one_hot(labels.long(), num_classes = nlabels)
        labels = labels.transpose(1,4).transpose(2,4).transpose(3,4)

    if loss_type == 'crossentropy':
        return losses.pixel_wise_cross_entropy_loss(labels_pred, labels)
    elif loss_type == 'dice':
        # labels_pred = [8,2,144,112,48]
        # labels = [8,2,144,112,48]
        return losses.dice_loss(labels_pred, labels)
    else:
        raise ValueError('Unknown loss type {}'.format(loss_type))


def do_eval(model,
            exp_config,
            images_set,
            labels_set,
            device: torch.device,
            training_data = True,
            label_hot_bool = False
            ):

    '''
    
    '''

    loss_ii = 0
    dice_ii = 0
    num_batches = 0
    
    model.eval()
    batch_size = exp_config.batch_size

    for batch in iterate_minibatches(images_set, labels_set, batch_size=batch_size, cut_t_slices=False, n_cut_slices=5):
        with torch.no_grad():
            inputs, labels = batch

            # From numpy.ndarray to tensors
            # Input (batch_size, x,y,z,channel_number)
            inputs = torch.from_numpy(inputs).transpose(1,4).transpose(2,4).transpose(3,4)
            # Input (batch_size, channell,x,y,z)

            # Labels (batch,size, x,y,z)
            labels = torch.from_numpy(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            if labels.shape[0] < batch_size:
                continue
            
            pred = model(inputs)

            loss_batch, dice_batch = evaluate_losses(
                pred,
                labels,
                exp_config.nlabels,
                exp_config.loss_type,
                label_hot_bool = label_hot_bool,
                )

            
            loss_ii += loss_batch
            dice_ii += dice_batch
            num_batches += 1
            
    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches
    logging.info('  Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))
    if training_data:
        wandb.log({"Average Trainning Loss":avg_loss, "Average Training Dice Score": avg_dice})

    else:
        wandb.log({"Average Evaluation Loss":avg_loss, "Average Evaluation Dice Score": avg_dice})



    return avg_loss, avg_dice

# Create session is the main function of the training script


SMOOTH = 1e-6

def IOU(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W x D shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W X D=> BATCH x H x W X D
    
    intersection = (outputs & labels).float().sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2,3))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

def train_model_debug(model: torch.nn.Module, 
                images_tr,
                labels_tr,
                images_val,
                labels_val,
                device: torch.device,
                optimizer: torch.optim,
                log_dir: str,
                config_exp):
    print("Training model...")
    print("Data augmentation ratio: {}".format(config_exp.da_ratio))
    print("Loss function: {}".format(config_exp.loss_type))
    print("Optimizer: {}".format(config_exp.optimizer_handle))
    print("Learning rate: {}".format(config_exp.learning_rate))
    print("Batch size: {}".format(config_exp.batch_size))
    print("Number of steps: {}".format(config_exp.steps))

    table_watch_train = wandb.Table(columns=["step", "image_number", "pred", "true"])
    table_watch_val = wandb.Table(columns=["step", "image_number", "pred", "true"])

    best_val_dice = 0
    step = 0
    image = images_tr[0]
    label = labels_tr[0]

    criterion = torch.nn.CrossEntropyLoss()
    for step in tqdm(range(config_exp.steps)):
        model.train

        inputs = torch.from_numpy(image)[None]
        # Input (batch_size, channell,x,y,z)
        inputs.transpose_(1,4).transpose_(2,4).transpose_(3,4)
        # Labels (batch,size, x,y,z)

        inputs = inputs.to(device)
        labels = torch.from_numpy(label)[None]
        labels = labels.to(device)

        
        inputs_hat = model(inputs)
        
        #loss = loss_function(inputs_hat, labels, config_exp.nlabels, config_exp.loss_type, labels_as_1hot = True)
        loss = criterion(inputs_hat, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging.info('step %d: training_loss = %.2f' % (step, loss))
        wandb.log({"step": step, "training_loss":loss})


        if step % 50 == 0:
                labels_dice = torch.nn.functional.one_hot(labels.long(), num_classes = config_exp.nlabels)
                labels_dice = labels_dice.transpose(1,4).transpose(2,4).transpose(3,4)
                _, dice_, _ = losses.compute_dice(inputs_hat, labels_dice)
                logging.info('step %d: training_dice = %.2f' % (step, dice_))
                ### Save images every 5 epoch (400)
                utils.make_dir_safely(log_dir + '/results/' + 'visualization/' + 'training/')
                #utils.make_dir_safely(log_dir + '/results/' + 'visualization/' + 'validation/')
                save_path = log_dir + '/results/' + 'visualization/' + 'training/'
                
                        
                prediction = F.softmax(inputs_hat, dim=1).argmax(dim = 1)
                np.save(save_path + f"pred_image_{step}.npy", prediction.detach().cpu().numpy())
                np.save(save_path + f"true_image_{step}.npy", labels.cpu().numpy())

            
                
                table_watch_train.add_data(step, step, wandb.Image(prediction[0,:,:,0].float()), wandb.Image(labels[0,:,:,0].float()))
                table_watch_train.add_data(step, step, wandb.Image(prediction[0,:,:,5].float()), wandb.Image(labels[0,:,:,5].float()))
                table_watch_train.add_data(step, step, wandb.Image(prediction[0,:,:,10].float()), wandb.Image(labels[0,:,:,10].float()))
    wandb.log({"Training table": table_watch_train})



def train_model(model: torch.nn.Module, 
                images_tr,
                labels_tr,
                images_val,
                labels_val,
                device: torch.device,
                optimizer: torch.optim,
                log_dir: str,
                config_exp):
    
    print("Training model...")
    print("Data augmentation ratio: {}".format(config_exp.da_ratio))
    print("Loss function: {}".format(config_exp.loss_type))
    print("Optimizer: {}".format(config_exp.optimizer_handle))
    print("Learning rate: {}".format(config_exp.learning_rate))
    print("Batch size: {}".format(config_exp.batch_size))
    print("Number of epochs: {}".format(config_exp.epochs))
    
    
    table_watch_train = wandb.Table(columns=["epoch", "image_number", "pred", "true", "input"])
    table_watch_val = wandb.Table(columns=["epoch", "image_number", "pred", "true", "input"])
    best_val_dice = 0
    #scheduler = CyclicLR(optimizer, base_lr = 5e-4 , max_lr = 1e-3, step_size_up=88, cycle_momentum=False )


    if config_exp.loss_type == "dice":
        label_hot_bool = False
    else:
        label_hot_bool = True
        
    

    step = 0

    for epoch in tqdm(range(config_exp.epochs)):
        print(" -------------------Epoch {}/{}   -------------------".format(epoch, config_exp.epochs))

                
        

        for batch in iterate_minibatches(images_tr, labels_tr, config_exp.batch_size, cut_t_slices=False, n_cut_slices=5):
            # Set model to training mode

            model.train()

            inputs, labels = batch

            # From numpy.ndarray to tensors

            # Input (batch_size, x,y,t,channel_number)
            inputs = torch.from_numpy(inputs)
            # Input (batch_size, channell,x,y,t)
            inputs.transpose_(1,4).transpose_(2,4).transpose_(3,4)
            # Labels (batch,size, x,y,t)

            inputs = inputs.to(device)
            labels = torch.from_numpy(labels).to(device)
            optimizer.zero_grad()
            
            inputs_hat = model(inputs)
            loss = loss_function(inputs_hat, labels, config_exp.nlabels, config_exp.loss_type, labels_as_1hot = label_hot_bool)
            
            loss.backward()
            optimizer.step()
            #wandb.log({ "lr": scheduler.get_last_lr()[0]})
            #scheduler.step()

        
            if (step) % exp_config.summary_writing_frequency == 0:                    
            #if step %20 ==0:
                    logging.info('step %d: training_loss = %.2f' % (step, loss))
                    wandb.log({"step": step, "training_loss":loss})
            #train_loss += loss.item() # * inputs.size(0)

            # ===========================
            # Compute the loss on the entire training set
            # ===========================

            #if (step) % exp_config.train_eval_frequency == 0:
            if step % 200 == 0:
                logging.info('Training Data Eval:')
                
                train_loss, train_dice = do_eval(model,
                        config_exp,
                        images_tr,
                        labels_tr,
                        device,
                        training_data = True,
                        label_hot_bool = label_hot_bool
                        )
                

            # ===========================
            # Save checkpoint
            # ===========================

            if step % config_exp.save_frequency == 0:
                save_path = os.path.join(log_dir, 'model_epoch_{}_step_{}.pth'.format(epoch, step))
                checkpoint(model, save_path)


            # ===========================
            # Evaluate on the validation set
            # ===========================
            #if (step) % exp_config.val_eval_frequency == 0:
            if step % 200 == 0:
                logging.info('Validation Data Eval:')
                utils.make_dir_safely(log_dir + '/results/')
                
                val_loss, val_dice = do_eval(model,
                        config_exp,
                        images_val,
                        labels_val,
                        device,
                        training_data = False,
                        label_hot_bool = label_hot_bool)
            if step % 200 == 0:
                ### Save images every 5 epoch (400)
                utils.make_dir_safely(log_dir + '/results/' + 'visualization/' + 'training/')
                utils.make_dir_safely(log_dir + '/results/' + 'visualization/' + 'validation/')

                
                save_results_visualization(model, config_exp, images_tr, labels_tr, device, os.path.join(log_dir + '/results/' + 'visualization/' + 'training/', f"step_{step}_"), table_watch=table_watch_train)
                save_results_visualization(model, config_exp, images_val, labels_val, device, os.path.join(log_dir + '/results/' + 'visualization/' + 'validation/', f"step_{step}_"), table_watch=table_watch_val)
            # ===========================
            # save model if the val dice is the best so far
            # ===========================

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_val_epoch = epoch
                best_val_step = step
                save_path = os.path.join(log_dir, 'model_best_{}_step_{}.pth'.format(best_val_epoch, best_val_step))
                checkpoint(model, save_path)
                logging.info('Best validation dice so far, saving model to %s' % save_path)

            step = step +1

    wandb.log({"Training table": table_watch_train})
    wandb.log({"Validation table": table_watch_val})

        
def save_results_visualization(model, config_exp, images_set, labels_set, device, save_path, table_watch = None):

    batch_size = config_exp.batch_size

    for n, batch in enumerate(iterate_minibatches(images_set, labels_set, batch_size=batch_size, cut_t_slices=False, n_cut_slices=5)):

        if n%10 == 0:

            with torch.no_grad():
                inputs, labels = batch

                # From numpy.ndarray to tensors
                # Input (batch_size, x,y,t,channel_number)
                inputs = torch.from_numpy(inputs).transpose(1,4).transpose(2,4).transpose(3,4)
                # Input (batch_size, channell,x,y,t)
                inputs = inputs.to(device)
                
                labels = torch.from_numpy(labels)

                if labels.shape[0] < batch_size:
                    continue
                
                logits = model(inputs)
                prediction = F.softmax(logits, dim=1).argmax(dim = 1)
                np.save(save_path + f"pred_image_{n}.npy", prediction.detach().cpu().numpy())
                np.save(save_path + f"true_image_{n}.npy", labels)
                np.save(save_path + f"input_image_{n}.npy", inputs.detach().cpu().numpy())

            
                epoch = save_path.split("/")[-1].split("_")[1]
                table_watch.add_data(epoch, n, wandb.Image(prediction[0,:,:,0].float()), wandb.Image(labels[0,:,:,0].float()), wandb.Image(inputs[0,0,:,:,0].float()))
                table_watch.add_data(epoch, n, wandb.Image(prediction[0,:,:,3].float()), wandb.Image(labels[0,:,:,3].float()), wandb.Image(inputs[0,0,:,:,3].float()))
                table_watch.add_data(epoch, n, wandb.Image(prediction[0,:,:,10].float()), wandb.Image(labels[0,:,:,10].float()), wandb.Image(inputs[0,0,:,:,10].float()))
                table_watch.add_data(epoch, n, wandb.Image(prediction[0,:,:,20].float()), wandb.Image(labels[0,:,:,20].float()), wandb.Image(inputs[0,0,:,:,20].float()))
                

def cut_z_slices(images, labels, n_cut):
    n_data = images.shape[0]
    index = np.arange(n_data)
    # We know we have 32 slices
    # First dim is the number of patients
    index_shaped = index.reshape(-1, 32)
    index_keep = index_shaped[:, n_cut:-n_cut].flatten()
    return images[index_keep], labels[index_keep]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    wandb_mode = "online" # online/ disabled
    with wandb.init(mode= wandb_mode,project="segmentation", name = exp_config.experiment_name, notes = "segmentation", tags =["Freiburg"]):

        # Load the data
        logging.info('============================================================')
        logging.info('Loading training data from: ' + sys_config.project_data_root)    
        data_tr = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_root,
                                                        idx_start = 0,
                                                        idx_end = 19,
                                                        train_test='train')
        images_tr = data_tr['images_train']
        labels_tr = data_tr['labels_train']
            
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

        logging.info('============================================================')
        logging.info('Loading validation data from: ' + sys_config.project_data_root)        
        data_vl = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_root,
                                                        idx_start = 20,
                                                        idx_end = 24,
                                                        train_test='validation')
        images_vl = data_vl['images_validation']
        labels_vl = data_vl['labels_validation']        
            
        logging.info('Shape of validation images: %s' %str(images_vl.shape))
        logging.info('Shape of validation labels: %s' %str(labels_vl.shape))
        logging.info('============================================================')
        
        if exp_config.nchannels == 1:
            logging.info('============================================================')
            logging.info('Only the phase x images (channel 1) will be used for the segmentation...')
            logging.info('============================================================')

        if exp_config.cut_z != 0:
            # We remove parts of the images in the z direction
            logging.info('============================================================')
            logging.info('Cutting the images in the z direction...')
            logging.info('============================================================')
            images_tr, labels_tr = cut_z_slices(images_tr, labels_tr, n_cut = exp_config.cut_z)
            images_vl, labels_vl = cut_z_slices(images_vl, labels_vl, n_cut = exp_config.cut_z)
            logging.info('============================================================')
            logging.info('Dimensions after cutting...')
            logging.info('============================================================')
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]
            logging.info('Shape of validation images: %s' %str(images_vl.shape))
            logging.info('Shape of validation labels: %s' %str(labels_vl.shape))



        # Create the experiment directory
        log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create the model
        model = exp_config.model_handle(in_channels=exp_config.nchannels, out_channels=exp_config.nlabels)
        model.to(device)


        # Create the optimizer
        optimizer = exp_config.optimizer_handle(model.parameters(), lr=exp_config.learning_rate)

        wandb.config.update({"experiment_name": exp_config.experiment_name, "data_augmentation": exp_config.da_ratio, "batch_size": exp_config.batch_size, "learning_rate": exp_config.learning_rate, "optimizer": exp_config.optimizer_handle, "model": exp_config.model_handle, "nchannels": exp_config.nchannels, "nlabels": exp_config.nlabels,
                             "epochs": exp_config.epochs, "loss": exp_config.loss_type, "z_cut":exp_config.cut_z })
        
        #"---------------------------------- DEBUGGING -------------------------------------"
        #logging.info('------------------------------------Debugging with smaller set--------------------------')
        #np.random.seed(0)
        #n_patients = 1  
        ##indexes_tr = np.sort(np.random.default_rng().choice(len(images_tr), size = n_patients *(32 - 2*exp_config.cut_z), replace = False))
        ##images_tr =images_tr[indexes_tr] 
        #images_tr = images_tr[np.arange(32 - 2*exp_config.cut_z)]
#
        ##labels_tr = labels_tr[indexes_tr]
        #labels_tr = labels_tr[np.arange(32 - 2*exp_config.cut_z)]
        #np.random.seed(0)
        #n_patients_val = 2
        #indexes_vl = np.sort(np.random.default_rng().choice(len(images_vl), size = n_patients_val *(32 - 2*exp_config.cut_z), replace = False))
        #images_vl = images_vl[indexes_vl]
        #np.random.seed(0)
        #labels_vl = labels_vl[indexes_vl]
        #logging.info('Shape of validation images: %s' %str(images_vl.shape))
        #logging.info('Shape of validation labels: %s' %str(labels_vl.shape))
        ## Print summary of model
        print(summary(exp_config.model_handle(exp_config.nchannels, exp_config.nlabels).to(device), torch.zeros(size = (exp_config.batch_size,exp_config.nchannels, 144,112,48) ).to(device)))
        train_model(model, images_tr, labels_tr, images_vl, labels_vl, device, optimizer, log_dir, exp_config)



def main_debug():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Load the data
    logging.info('============================================================')
    logging.info('Loading training data from: ' + sys_config.project_data_root)    
    data_tr = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_root,
                                                    idx_start = 0,
                                                    idx_end = 19,
                                                    train_test='train')
    images_tr = data_tr['images_train']
    labels_tr = data_tr['labels_train']
        
    logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

    logging.info('============================================================')
    logging.info('Loading validation data from: ' + sys_config.project_data_root)        
    data_vl = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_root,
                                                    idx_start = 20,
                                                    idx_end = 24,
                                                    train_test='validation')
    images_vl = data_vl['images_validation']
    labels_vl = data_vl['labels_validation']        
        
    logging.info('Shape of validation images: %s' %str(images_vl.shape))
    logging.info('Shape of validation labels: %s' %str(labels_vl.shape))
    logging.info('============================================================')
    
    if exp_config.nchannels == 1:
        logging.info('============================================================')
        logging.info('Only the first phase images (channel 1) will be used for the segmentation...')
        logging.info('============================================================')
        # Create the experiment directory
    log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the model
    model = exp_config.model_handle(in_channels=exp_config.nchannels, out_channels=exp_config.nlabels)
    model.to(device)

    # Create the optimizer
    optimizer = exp_config.optimizer_handle(model.parameters(), lr=exp_config.learning_rate)

    np.random.seed(0)
    n_patients = 1  
    indexes_tr = np.sort(np.random.default_rng().choice(len(images_tr), size =1, replace = False))
    images_tr_debug =images_tr[indexes_tr] 
    labels_tr_debug = labels_tr[indexes_tr]
    # Repeat the same image several times 
    images_tr_debug = np.repeat(images_tr_debug, 32, axis =0)
    labels_tr_debug = np.repeat(labels_tr_debug, 32, axis =0)
    images_tr_debug_time = images_tr_debug[:,:,:,7,:] 
    labels_tr_debug_time = labels_tr_debug[:,:,:,7]
    images_tr_debug = np.repeat(images_tr_debug_time[:,:,:,np.newaxis,:], 48, axis =3)
    labels_tr_debug = np.repeat(labels_tr_debug_time[:,:,:,np.newaxis], 48, axis =3)

    np.random.seed(0)
    n_patients_val = 1
    indexes_vl = np.sort(np.random.default_rng().choice(len(images_vl), size = n_patients_val *32, replace = False))
    images_vl_debug = images_vl[indexes_vl]
    np.random.seed(0)
    labels_vl_debug = labels_vl[indexes_vl]
    

    logging.info('Shape of training images: %s' %str(images_tr_debug.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    logging.info('Shape of training labels: %s' %str(labels_tr_debug.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]
    logging.info('Shape of validation images: %s' %str(images_vl_debug.shape))
    logging.info('Shape of validation labels: %s' %str(labels_vl_debug.shape))
    criterion = torch.nn.CrossEntropyLoss()
    for e in tqdm(range(400)):
        train_loss = 0
        n_batches = 0
        for batch in iterate_minibatches(images_tr_debug, labels_tr_debug, batch_size= 8, cut_t_slices=False, n_cut_slices=5):
            model.train()
            x, y = batch

            # From numpy.ndarray to tensors

            # Input (batch_size, x,y,t,channel_number)
            x = torch.from_numpy(x)
            # Input (batch_size, channell,x,y,t)
            x.transpose_(1,4).transpose_(2,4).transpose_(3,4)
            # Labels (batch,size, x,y,t)

            x = x.to(device)
            y = torch.from_numpy(y).to(device)
            
            optimizer.zero_grad()
            
            x_logits = model(x)
            loss = criterion(x_logits, y.long())
            train_loss += loss.item()
            #one_hot_labels= torch.nn.functional.one_hot(y.to(torch.int64), num_classes = 2).transpose(1,4).transpose(2,4).transpose(3,4)
            #_, dice_train, _ =losses.compute_dice(x_logits, one_hot_labels)
            #print("dice_mean training", dice_train.item())
            loss.backward()
            optimizer.step()
            n_batches +=1
        print("train_loss_divided by number of batches", train_loss/n_batches, "train_loss summed", train_loss)
        
        if e % 20 == 0:
            np.save(f"/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/testing_batches/cut_channel_1_debug_01/images_{e}.npy", x.cpu().numpy())
            
            np.save(f"/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/testing_batches/cut_channel_1_debug_01/prediction_{e}.npy", x_logits.detach().cpu().numpy())
            np.save(f"/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/testing_batches/cut_channel_1_debug_01/labels_{e}.npy", y.cpu().numpy())

            """
            val_loss = 0
            n_batches = 0
            for batch in iterate_minibatches(images_vl_debug, labels_vl_debug, batch_size= 8):
                model.eval()
                x, y = batch

                # From numpy.ndarray to tensors

                # Input (batch_size, x,y,t,channel_number)
                x = torch.from_numpy(x)
                # Input (batch_size, channell,x,y,t)
                x.transpose_(1,4).transpose_(2,4).transpose_(3,4)
                # Labels (batch,size, x,y,t)

                x = x.to(device)
                y = torch.from_numpy(y).to(device)
                
                x_logits = model(x)
                one_hot_labels = torch.nn.functional.one_hot(y.to(torch.int64), num_classes = 2).transpose(1,4).transpose(2,4).transpose(3,4)
                _, dice_val, _ =losses.compute_dice(x_logits, one_hot_labels)
                print("dice_mean validation", dice_val.item())
                loss = criterion(x_logits, y.long())
                val_loss += loss.item()
                n_batches +=1
            print("val_loss_divided by number of batches", val_loss/n_batches, "val loss summed", val_loss)
            """



if __name__ == '__main__':
    main()
