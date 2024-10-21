# activate the seg_net environment 
import logging
import os.path
import yaml
import numpy as np
import torch
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation')
import config.system as sys_config
sys.path.append(sys_config.project_code_root + '/src')
from helpers import data_freiburg_numpy_to_hdf5, losses
import wandb
import argparse
from tqdm import tqdm
import h5py
from pytorch_model_summary import summary
from helpers.utils import make_dir_safely, generate_experiment_name, checkpoint, iterate_minibatches, setup_scheduler, cut_z_slices, save_results_visualization
from models import model_zoo


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

    _, dice_, mean_fg_dice = losses.compute_dice(labels_pred, labels)

    return loss_, dice_, mean_fg_dice


def loss_function(labels_pred, labels, nlabels, loss_type, labels_as_1hot = False, weights = None):
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
        return losses.pixel_wise_cross_entropy_loss(labels_pred, labels, weights = weights)
    elif loss_type == 'dice':
        # labels_pred = [8,2,144,112,48]
        # labels = [8,2,144,112,48]
        return losses.dice_loss(labels_pred, labels, weights = weights)
    else:
        raise ValueError('Unknown loss type {}'.format(loss_type))


def do_eval(model,
            config,
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
    fg_dice_ii = 0
    num_batches = 0
    
    model.eval()
    batch_size = config["batch_size"]

    for batch in iterate_minibatches(images_set, labels_set, batch_size, config):
        with torch.no_grad():
            inputs, labels, _ = batch

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

            loss_batch, dice_batch, fg_dice_batch = evaluate_losses(
                pred,
                labels,
                config["nlabels"],
                config["loss_type"],
                label_hot_bool = label_hot_bool,
                )

            
            loss_ii += loss_batch
            dice_ii += dice_batch
            fg_dice_ii += fg_dice_batch
            num_batches += 1
            
    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches
    avg_fg_dice = fg_dice_ii / num_batches
    logging.info('  Average loss: %0.04f, average dice: %0.04f, average fg dice: %0.04f' % (avg_loss, avg_dice, avg_fg_dice))
    if training_data:
        wandb.log({"Average Trainning Loss":avg_loss, "Average Training Dice Score": avg_dice, "Average Training FG Dice Score": avg_fg_dice})

    else:
        wandb.log({"Average Evaluation Loss":avg_loss, "Average Evaluation Dice Score": avg_dice, "Avegage Evaluation FG Dice Score": avg_fg_dice})



    return avg_loss, avg_dice


def train_model(model: torch.nn.Module,
                images_tr,
                labels_tr,
                images_val,
                labels_val,
                device: torch.device,
                optimizer: torch.optim,
                exp_dir: str,
                config):
    """
    Function to train the model
    :param model: model to be trained
    :param images_tr: training images
    :param labels_tr: training labels
    :param images_val: validation images
    :param labels_val: validation labels
    :param device: device to run the model
    :param optimizer: optimizer
    :param exp_dir: experiment directory
    :param config: configuration file
    """

    logging.info("Training model...")
    logging.info("Data augmentation ratio: {}".format(config["da_ratio"]))
    logging.info("Loss function: {}".format(config["loss_type"]))
    logging.info("Optimizer: {}".format(config["optimizer_handle"]))
    logging.info("Learning rate: {}".format(config["learning_rate"]))
    logging.info("Batch size: {}".format(config["batch_size"]))
    logging.info("Number of epochs: {}".format(config["epochs"]))
    logging.info("Pixel_weight added to not have 0: {}".format(config["add_pixels_weight"]))
    logging.info('Use validation set: {}'.format(config["with_validation"]))
    if config["REPRODUCE"]:
        logging.info('Seed: {}'.format(config["SEED"]))

    # Depending on if we continue a run or not, we log that
    if config["continue_run"]:
        logging.info("Continuing run from previous checkpoint")

    # If scheduler is used, log the scheduler
    if config["scheduler"]['use_scheduler']:
        logging.info(f'Using scheduler: {config["scheduler"]["type"]}')
        logging.info(f'Scheduler parameters: {config["scheduler"]}')



    wandb.watch(model, log="all", log_freq=10)
    wandb.define_metric("best_validation", summary="max")
    table_watch_train = wandb.Table(columns=["epoch", "image_number", "pred", "true", "input"])
    table_watch_val = wandb.Table(columns=["epoch", "image_number", "pred", "true", "input"])
    best_val_dice = 0
    # Optionally set up scheduler based on the config_exp
    scheduler = setup_scheduler(optimizer, config)

    # If the loss type is dice, we do not use one hot encoding (for crossentropy we do)
    if config["loss_type"] == "crossentropy":
        label_hot_bool = True
    else:
        label_hot_bool = False

    step = 0

    for epoch in tqdm(range(config["epochs"])):
        logging.info(" -------------------Epoch {}/{}   -------------------".format(epoch, config["epochs"]))

        for batch in iterate_minibatches(images_tr, labels_tr, config["batch_size"], config):
            # Set model to training mode
            model.train()
            inputs, labels, weights = batch
            weights = torch.from_numpy(weights).to(device)
            inputs = torch.from_numpy(inputs) # Input (batch_size, x,y,t,channel_number)
            inputs.transpose_(1,4).transpose_(2,4).transpose_(3,4) # Input (batch_size, channell,x,y,t)
            inputs = inputs.to(device)
            labels = torch.from_numpy(labels).to(device)
            
            optimizer.zero_grad()
            inputs_hat = model(inputs)
            loss = loss_function(inputs_hat, labels, config["nlabels"], config["loss_type"], labels_as_1hot=label_hot_bool, weights=None)
            loss.backward()
            optimizer.step()

            # See if scheduler is used and step the scheduler
            if scheduler is not None:
                scheduler.step()
                wandb.log({ "lr": scheduler.get_last_lr()[0]})

            if (step) % config["summary_writing_frequency"] == 0:
                logging.info('step %d: training_loss = %.2f' % (step, loss))
                wandb.log({"step": step, "training_loss": loss})
            
            # Compute the loss on the entire training set
            if (step) % config["train_eval_frequency"] == 0:
                logging.info('Training data evaluation:')
                _, _ = do_eval(model,
                            config,
                            images_tr,
                            labels_tr,
                            device,
                            training_data=True,
                            label_hot_bool=label_hot_bool)
            
            # Save checkpoint
            if step % config["save_frequency"] == 0:
                save_path = os.path.join(exp_dir, 'model_epoch_{}_step_{}.pth'.format(epoch, step))
                checkpoint(model, save_path)

            # Evaluate on the validation set
            if config["with_validation"]:
                if (step) % config["val_eval_frequency"] == 0:
                    logging.info('Validation data evaluation:')
                    make_dir_safely(exp_dir + '/results/')
                    val_loss, val_dice = do_eval(model,
                            config,
                            images_val,
                            labels_val,
                            device,
                            training_data=False,
                            label_hot_bool=label_hot_bool)

                if step % 400 == 0:
                    # Save images every 5 epoch (400)
                    make_dir_safely(exp_dir + '/results/' + 'visualization/' + 'training/')
                    make_dir_safely(exp_dir + '/results/' + 'visualization/' + 'validation/')
                    save_results_visualization(model, config, images_tr, labels_tr, device, os.path.join(exp_dir + '/results/' + 'visualization/' + 'training/', f"step_{step}_"), table_watch=table_watch_train)
                    save_results_visualization(model, config, images_val, labels_val, device, os.path.join(exp_dir + '/results/' + 'visualization/' + 'validation/', f"step_{step}_"), table_watch=table_watch_val)

                # Save model if the val dice is the best so far
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    best_val_epoch = epoch
                    best_val_step = step
                    save_path = os.path.join(exp_dir, 'model_best_{}_step_{}.pth'.format(best_val_epoch, best_val_step))
                    checkpoint(model, save_path)
                    logging.info('Best validation dice so far, saving model to %s' % save_path)
                    wandb.log({"best_validation": best_val_dice})
            else:
                if step % config["save_frequency"] == 0:
                    make_dir_safely(exp_dir + '/results/' + 'visualization/' + 'training/')
                    save_results_visualization(model, config, images_tr, labels_tr, device, os.path.join(exp_dir + '/results/' + 'visualization/' + 'training/', f"step_{step}_"), table_watch=table_watch_train)

            step = step + 1
    wandb.log({"Training table": table_watch_train})
    wandb.log({"Validation table": table_watch_val})


if __name__ == '__main__':

    
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train the segmentation network')
    parser.add_argument('--config_path', type=str, help='Path to configuration YAML file')

    # Parse the arguments
    args = parser.parse_args()
    # Load the configuration file
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if config["REPRODUCE"]:
        SEED = config["SEED"]
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Get the slurm job id
    config['AAslurm_job_id'] = os.environ.get("SLURM_JOB_ID")
    model_mapping = {
    "UNet": model_zoo.UNet,
    
}

    # Ensure compatibility between configurations
    if config["use_adaptive_batch_norm"] and config["train_with_bern"]:
        raise ValueError(
            "Invalid configuration: 'use_adaptive_batch_norm' cannot be True while 'train_with_bern' is True. "
            "Set 'train_with_bern' to False or 'use_adaptive_batch_norm' to False."
        )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {}".format(device))

    if config["use_wandb"]:
        wandb_mode = "online"
    else:
        wandb_mode = "disabled"
    

    # Add tags to the experiment based on config file
    wandb_tags = [f'epochs_{config["epochs"]}', f'loss_function_{config["loss_type"]}', f'{config["train_file_name"]}', f'use_adaptive_batch_norm_{config["use_adaptive_batch_norm"]}', f'train_with_bern_{config["train_with_bern"]}', f'use_saved_model_{config["use_saved_model"]}'
                  , f'defrozen_conv_blocks_{config["defrozen_conv_blocks"]}', f'only_w_bern_{config["only_w_bern"]}', f'{config["val_file_name"]}',
                  f'reproduce_{config["REPRODUCE"]}', f'seed_{config["SEED"]}', f'{config["AAslurm_job_id"]}']
    
    
    experiment_name = generate_experiment_name(config)
    config['experiment_name'] = experiment_name


    with wandb.init(mode= wandb_mode,project="3D_segmentation_aorta", name = experiment_name, notes = "segmentation", tags =wandb_tags):
        # Create the model
        model = model_mapping[config["model_handle"]](in_channels=config["nchannels"], out_channels=config["nlabels"])
        model.to(device)
        
        # We have several ways of training the segmentation network:
        # 1. Training a model only with the Freiburg data (use_saved_model = False and train_with_bern = False)
        # 2. Training with Bern data from scratch without Freiburg (use_saved_model = False, train_with_bern = True, only_w_bern = True)
        # 3. Finetuning with batch normalization for Bern data with saved model (use_saved_model = True, use_adaptive_batch_norm = True)
        # 4. Traininig with Bern and Freiburg data (False = True, train_with_bern = True, only_w_bern = False)
        # 5. Finetune with Bern but with all layers (use_saved_model = True, train_with_bern = True)
        
        if not config["use_saved_model"] and not config["train_with_bern"]:
            # Load the data
            logging.info('============================================================')
            logging.info('Loading training data from: ' + sys_config.project_data_freiburg_root)    
            data_tr = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_freiburg_root,
                                                            idx_start = 0,
                                                            idx_end = 19,
                                                            train_test='train')
            images_tr = data_tr['images_train']
            labels_tr = data_tr['labels_train']
                
            logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
            logging.info('Shape of training labels: %s' %str(labels_tr.shape))

            logging.info('============================================================')
            logging.info('Loading validation data from: ' + sys_config.project_data_freiburg_root)
            data_vl = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_freiburg_root,
                                                            idx_start = 20,
                                                            idx_end = 24,
                                                            train_test='validation')
            images_vl = data_vl['images_validation']
            labels_vl = data_vl['labels_validation']

            logging.info('Shape of validation images: %s' %str(images_vl.shape))
            logging.info('Shape of validation labels: %s' %str(labels_vl.shape))
            logging.info('============================================================')

            if config["cut_z"]:
                # Values are either 0 or 3 and this function works for the Freiburg dataset
                # We remove parts of the images in the z direction
                logging.info('============================================================')
                logging.info('Cutting the images in the z direction...')
                logging.info('============================================================')
                images_tr, labels_tr = cut_z_slices(images_tr, labels_tr, freiburg = True)
                images_vl, labels_vl = cut_z_slices(images_vl, labels_vl, freiburg = True)
                logging.info('============================================================')
                logging.info('Dimensions after cutting...')
                logging.info('============================================================')
                logging.info('Shape of training images: %s' %str(images_tr.shape))
                logging.info('Shape of training labels: %s' %str(labels_tr.shape))
                logging.info('Shape of validation images: %s' %str(images_vl.shape))
                logging.info('Shape of validation labels: %s' %str(labels_vl.shape))
        
        if ((not config["use_saved_model"]) and (config["train_with_bern"]) and (config["only_w_bern"])):
            # Training with Bern data from scratch without Freiburg
            logging.info('============================================================')
            logging.info('Training with Bern data from scratch without Freiburg')
            data_tr = h5py.File(sys_config.project_data_bern_root + f'/{config["train_file_name"]}.hdf5','r')
            data_vl = h5py.File(sys_config.project_data_bern_root + f'/{config["val_file_name"]}.hdf5','r')
            images_tr = data_tr['images_train'][:]
            labels_tr = data_tr['labels_train'][:]
            images_vl = data_vl['images_validation'][:]
            labels_vl = data_vl['labels_validation'][:]
            logging.info('Shape of training images: %s' %str(images_tr.shape))
            logging.info('Shape of training labels: %s' %str(labels_tr.shape))
            logging.info('Shape of validation images: %s' %str(images_vl.shape))
            logging.info('Shape of validation labels: %s' %str(labels_vl.shape))

        
        # Load the saved models if that's what we use
        if config["use_saved_model"]:
            logging.info('============================================================')
            logging.info('Loading model from: ' + config["experiment_name_saved_model"])
            model_path = os.path.join(sys_config.log_root, config["experiment_name_saved_model"])
            best_model_path = os.path.join(model_path, list(filter(lambda x: 'best' in x, os.listdir(model_path)))[-1])
            model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Use finetuning with batch normalization for Bern data
        if config["use_adaptive_batch_norm"] and config["use_saved_model"]:
            logging.info("Using adaptive batch norm")

            # If the layer is a batch norm then we want to keep it trainable but the others not
            #TODO: Improve this 16.10.2024
            if not config["defrozen_conv_blocks"]:
                for name, param in model.named_parameters():
                    if 'bn' not in name:
                        param.requires_grad = False
            else:
                logging.info("Defreezing conv blocks")
                for name, param in model.named_parameters():
                    if ('bn' not in name) and ('upconv' in name) ^ ('conv3' in name) ^ ('conv4' in name) ^ ('conv5' in name) ^ ('conv6' in name):
                        param.requires_grad = False

            logging.info("Loading Bern data... ")
            data_tr = h5py.File(sys_config.project_data_bern_root + f'/{config["train_file_name"]}.hdf5','r')
            data_vl = h5py.File(sys_config.project_data_bern_root + f'/{config["val_file_name"]}.hdf5','r')
            images_tr = data_tr['images_train'][:]
            labels_tr = data_tr['labels_train'][:]
            images_vl = data_vl['images_validation'][:]
            labels_vl = data_vl['labels_validation'][:]
            logging.info('Shape of Bern training images: %s' %str(images_tr.shape))
            logging.info('Shape of Bern training labels: %s' %str(labels_tr.shape))
            logging.info('Shape of Bern validation images: %s' %str(images_vl.shape))
            logging.info('Shape of Bern validation labels: %s' %str(labels_vl.shape))

            if config["cut_z"]:
                logging.info('============================================================')
                logging.info('Cutting the images in the z direction...')
                logging.info('============================================================')
                keep_indices_tr = np.where(data_tr['alias'][:] ==0)[0]
                keep_indices_vl = np.where(data_vl['alias'][:] ==0)[0]
                images_tr = images_tr[keep_indices_tr]
                labels_tr = labels_tr[keep_indices_tr]
                images_vl = images_vl[keep_indices_vl]
                labels_vl = labels_vl[keep_indices_vl]
                logging.info('============================================================')
                logging.info('Dimensions after cutting...')
                logging.info('============================================================')
                logging.info('Shape of Bern training images: %s' %str(images_tr.shape))
                logging.info('Shape of Bern training labels: %s' %str(labels_tr.shape))
                logging.info('Shape of Bern validation images: %s' %str(images_vl.shape))
                logging.info('Shape of Bern validation labels: %s' %str(labels_vl.shape))

        # Here we train with Bern data and Freiburg data
        if ((config["train_with_bern"]) and (not config["use_saved_model"]) and (not config["only_w_bern"])):
            logging.info("Loading Bern data... and appending to the Freiburg data...")
            bern_tr = h5py.File(sys_config.project_data_bern_root + f'/{config["train_file_name"]}.hdf5','r')
            bern_vl = h5py.File(sys_config.project_data_bern_root + f'/{config["val_file_name"]}.hdf5','r')
            data_tr = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_freiburg_root,
                                                            idx_start = 0,
                                                            idx_end = 19,
                                                            train_test='train')
            images_tr = data_tr['images_train']
            labels_tr = data_tr['labels_train']
            data_vl = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_freiburg_root,
                                                            idx_start = 20,
                                                            idx_end = 24,
                                                            train_test='validation')
            images_vl = data_vl['images_validation']
            labels_vl = data_vl['labels_validation']
            bern_images_tr = bern_tr['images_train']
            bern_labels_tr = bern_tr['labels_train']
            bern_images_vl = bern_vl['images_validation']
            bern_labels_vl = bern_vl['labels_validation']
            logging.info('Shape of Bern training images: %s' %str(bern_images_tr.shape))
            logging.info('Shape of Bern training labels: %s' %str(bern_labels_tr.shape))
            logging.info('Shape of Bern validation images: %s' %str(bern_images_vl.shape))
            logging.info('Shape of Bern validation labels: %s' %str(bern_labels_vl.shape))

            if config["cut_z"]:
                logging.info('============================================================')
                logging.info('Cutting the images in the z direction...')
                logging.info('============================================================')
                keep_indices_tr = np.where(bern_tr['alias'][:] ==0)[0]
                keep_indices_vl = np.where(bern_vl['alias'][:] ==0)[0]
                bern_images_tr = bern_images_tr[keep_indices_tr]
                bern_labels_tr = bern_labels_tr[keep_indices_tr]
                bern_images_vl = bern_images_vl[keep_indices_vl]
                bern_labels_vl = bern_labels_vl[keep_indices_vl]
                logging.info('============================================================')
                logging.info('Dimensions after cutting...')
                logging.info('============================================================')
                logging.info('Shape of Bern training images: %s' %str(bern_images_tr.shape))
                logging.info('Shape of Bern training labels: %s' %str(bern_labels_tr.shape))
                logging.info('Shape of Bern validation images: %s' %str(bern_images_vl.shape))
                logging.info('Shape of Bern validation labels: %s' %str(bern_labels_vl.shape))

            images_tr = np.concatenate([images_tr[:],bern_images_tr[:]], axis=0)
            labels_tr = np.concatenate([labels_tr[:],bern_labels_tr[:]], axis=0)
            images_vl = np.concatenate([images_vl[:],bern_images_vl[:]], axis=0)
            labels_vl = np.concatenate([labels_vl[:],bern_labels_vl[:]], axis=0)
            logging.info("Done loading Bern data...")

        if ((config["use_saved_model"]) and (config["train_with_bern"])):
            # Here we finetune the model with Bern data using all the layers
            logging.info("Loading Bern data... ")
            logging.info("Finetuning the model with Bern data...")
            data_tr = h5py.File(sys_config.project_data_bern_root + f'/{config["train_file_name"]}.hdf5','r')
            data_vl = h5py.File(sys_config.project_data_bern_root + f'/{config["val_file_name"]}.hdf5','r')
            images_tr = data_tr['images_train'][:]
            labels_tr = data_tr['labels_train'][:]
            images_vl = data_vl['images_validation'][:]
            labels_vl = data_vl['labels_validation'][:]

            if config["cut_z"]:
                logging.info('============================================================')
                logging.info('Cutting the images in the z direction...')
                logging.info('============================================================')
                keep_indices_tr = np.where(data_tr['alias'][:] ==0)[0]
                keep_indices_vl = np.where(data_vl['alias'][:] ==0)[0]
                images_tr = images_tr[keep_indices_tr]
                labels_tr = labels_tr[keep_indices_tr]
                images_vl = images_vl[keep_indices_vl]
                labels_vl = labels_vl[keep_indices_vl]
                logging.info('============================================================')
                logging.info('Shapes after cutting...')
                logging.info('============================================================')

        if config["nchannels"] == 1:
            logging.info('============================================================')
            logging.info('Only the phase x images (channel 1) will be used for the segmentation...')
            logging.info('============================================================')

        # Create the experiment directory
        exp_dir = os.path.join(sys_config.log_experiments_root, experiment_name)
        make_dir_safely(exp_dir)

        # Create the optimizer
        # It is created after the model as some layers may be set to require_grad False
        if config["optimizer_handle"] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        elif config["optimizer_handle"] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), betas=config["betas"])
        else:
            raise ValueError("Invalid optimizer handle, select either Adam or AdamW")
        
        wandb.config.update({"experiment_name": experiment_name, "with_validation": config["with_validation"],"data_augmentation": config["da_ratio"], "batch_size": config["batch_size"], "learning_rate": config["learning_rate"], "optimizer": config["optimizer_handle"], "betas_if_adamW":config["betas"], "model": config["model_handle"], "nchannels": config["nchannels"], "nlabels": config["nlabels"],
                                    "epochs": config["epochs"], "loss": config["loss_type"], "z_cut":config["cut_z"], "use_bern_data":config["train_with_bern"], "defrozen_conv_blocks": config["defrozen_conv_blocks"], "adaptive_batch_norm": config["use_adaptive_batch_norm"], "pixel_pad_weight": config["add_pixels_weight"]})
        
        logging.info('Shape of FINAL training images: %s' %str(images_tr.shape))
        logging.info('Shape of FINAL training labels: %s' %str(labels_tr.shape))
        logging.info('Shape of FINAL validation images: %s' %str(images_vl.shape))
        logging.info('Shape of FINAL validation labels: %s' %str(labels_vl.shape))
        print(summary(model_mapping[config["model_handle"]](config["nchannels"], config["nlabels"]).to(device), torch.zeros(size = (config["batch_size"],config["nchannels"], 144,112,48) ).to(device)))
        train_model(model, images_tr, labels_tr, images_vl, labels_vl, device, optimizer, exp_dir, config)
