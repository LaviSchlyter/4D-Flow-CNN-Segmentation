import os
import glob
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import morphology
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import torch
import datetime
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, StepLR
    
# ===================================================
# ===================================================
def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''
    print(folder)

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):
        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))

# ==================================================================
# Save checkpoints
# ==================================================================

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
# ==========================================        
# function to normalize the input arrays (intensity and velocity) to a range between 0 to 1.
# magnitude normalization is a simple division by the largest value.
# velocity normalization first calculates the largest magnitude velocity vector
# and then scales down all velocity vectors with the magnitude of this vector.
# ==========================================        
def normalize_image(image):

    # ===============    
    # initialize with zeros
    # ===============
    normalized_image = np.zeros((image.shape))
    
    # ===============
    # normalize magnitude channel
    # ===============
    normalized_image[...,0] = image[...,0] / np.amax(image[...,0])
    
    # ===============
    # normalize velocities
    # ===============
    
    # extract the velocities in the 3 directions
    velocity_image = np.array(image[...,1:4])
    
    # denoise the velocity vectors
    velocity_image_denoised = gaussian_filter(velocity_image, 0.5)
    
    # compute per-pixel velocity magnitude    
    velocity_mag_image = np.linalg.norm(velocity_image_denoised, axis=-1)
    
    # find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    vpercentile = np.percentile(velocity_mag_image, 95)    
    normalized_image[...,1] = velocity_image_denoised[...,0] / vpercentile
    normalized_image[...,2] = velocity_image_denoised[...,1] / vpercentile
    normalized_image[...,3] = velocity_image_denoised[...,2] / vpercentile  
  
    return normalized_image


# ==================================================================    
# ==================================================================    
def make_dir_safely(dirname):
    # directory = os.path.dirname(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def generate_experiment_name(config):
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
        # Tags based on configuration
        da_tag = f'da_{config["da_ratio"]}'
        cutz_tag = '_cutz' if config['cut_z'] else ''
        channel_tag = f'nchan{config["nchannels"]}'
        loss_tag = f'loss_{config["loss_type"]}'
        epoch_tag = f'e{config["epochs"]}'
        batch_tag = f'bs{config["batch_size"]}'
        lr_tag = f'lr{config["learning_rate"]}'
        full_run_freiburg = '_full_run_freiburg' if not config['use_saved_model'] and not config['train_with_bern'] else ''
        adaptive_bn_tag = 'adBN' if config['use_adaptive_batch_norm'] else ''
        w_validation_tag = '_w_val' if config['with_validation'] else ''

        # Notes based on Bern and saved model
        note = ''
        if config['train_with_bern']:
            note = '_tr_only_w_labels' if 'only_w_labels' in config['train_file_name'] else '_tr_size_40'
        if config['only_w_bern']:
            note += '_only_w_bern'
        if config['use_saved_model']:
            note += '_finetune'
        if not config['use_saved_model'] and config['train_with_bern'] and not config['only_w_bern']:
            note += '_bern_and_freiburg'

        

        # Construct the experiment name
        experiment_name = f'{timestamp}_{da_tag}_{channel_tag}_{loss_tag}{cutz_tag}_{epoch_tag}_{batch_tag}_{lr_tag}_{adaptive_bn_tag}{full_run_freiburg}{w_validation_tag}{note}'
        # Add optimizer information
        optimizer_tag = f'opt_{config["optimizer_handle"]}'
        experiment_name += f'_{optimizer_tag}'

        # Add scheduler information if applicable
        if config['scheduler']['use_scheduler']:
            scheduler_type = config['scheduler']['type']
            experiment_name += f'_sched_{scheduler_type}'
        if config['REPRODUCE']:
            experiment_name += f'_seed_{config["SEED"]}'
        return experiment_name

# Visualize results of model throughout training
def save_results_visualization(model, config, images_set, labels_set, device, save_path, table_watch = None):

    batch_size = config["batch_size"]

    for n, batch in enumerate(iterate_minibatches(images_set, labels_set, batch_size=batch_size, config = config)):

        if n%100 == 0:

            with torch.no_grad():
                inputs, labels, _ = batch

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
                
# ==================================================================
# Iterate over mini-batches
# ==================================================================

def iterate_minibatches(images, labels, batch_size, config):
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
        
        
        # ===========================
        # check if the velocity fields are to be used for the segmentation...
        # ===========================
        if config['nchannels'] == 1:
            X = X[..., 1:2]
    
        # ===========================
        # augment the batch            
        # ===========================
        if config['da_ratio'] > 0.0 and config['nchannels'] == 1:
            X, y = augment_data(X,
                                      y,
                                      data_aug_ratio = config['da_ratio'])
            
        weights = (y.sum(axis = (1,2,3)) + config['add_pixels_weight'])/(y.shape[1]*y.shape[2]*y.shape[3])
        
        yield X, y, weights


# ==================================================================
# Schedule learning rate
# ==================================================================

def setup_scheduler(optimizer, config):
    scheduler = None
    if config['scheduler']['use_scheduler']:
        scheduler_type = config['scheduler']['type']
        
        if scheduler_type == 'StepLR':
            scheduler = StepLR(
                optimizer, 
                step_size=config['scheduler']['step_size'], 
                gamma=config['scheduler']['gamma']
            )
        elif scheduler_type == 'ExponentialLR':
            scheduler = ExponentialLR(
                optimizer, 
                gamma=config['scheduler']['gamma']
            )
        elif scheduler_type == 'CyclicLR':
            scheduler = CyclicLR(
                optimizer, 
                base_lr=config['scheduler']['base_lr'], 
                max_lr=config['scheduler']['max_lr'], 
                mode=config['scheduler']['mode'], 
                cycle_momentum=config['scheduler'].get('cycle_momentum', False)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

# ==================================================================
# Cut the z-axis
# ==================================================================
def cut_z_slices(images, labels, freiburg = False):
    if freiburg:
        n_data = images.shape[0]
        index = np.arange(n_data)
        # We know we have 32 slices (only valid for Freuburg data)
        # First dim is the number of patients
        index_shaped = index.reshape(-1, 32)
        index_keep = index_shaped[:, 3:-3].flatten()
        return images[index_keep], labels[index_keep]
    else:
        # We know we have 40 slices (only valid for Bern data)
        n_data = images.shape[0]
        index = np.arange(n_data)
        index_shaped = index.reshape(-1, 40)
        index_keep = index_shaped[:, 3:-3].flatten()
        return images[index_keep], labels[index_keep]
def crop_or_pad_Bern_all_slices(data, new_shape):
    #processed_data = np.zeros(new_shape)
    
    # axis 0 is the x-axis and we crop from top since aorta is at the bottom
    # axis 1 is the y-axis and we pad 
    # The axis two we leave since it'll just be the batch dimension
    delta_axis0 = data.shape[0] - new_shape[0]
    delta_axis1 = data.shape[1] - new_shape[1]
    if len(new_shape) == 5: # Image (x,y,None, t,c) - the z will be batch and will vary doesn't need to be equal
        processed_data = np.zeros((new_shape[0], new_shape[1], data.shape[2], new_shape[3], new_shape[4]))
        if delta_axis1 <= 0:
        # The x is always cropped, y padded
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,...]
        else:
            # x croped and y cropped equally either way
            processed_data[:, :,:,:data.shape[3],... ] = data[delta_axis0:, (delta_axis1//2):-(delta_axis1//2),...]


    if len(new_shape) == 4: # Label
        processed_data = np.zeros((new_shape[0], new_shape[1], data.shape[2], new_shape[3]))
        # The x is always cropped, y always padded
        if delta_axis1 <= 0:
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,...]
        else:
            # x croped and y cropped equally either way
            processed_data[:, :,:,:data.shape[3],... ] = data[delta_axis0:, (delta_axis1//2):-(delta_axis1//2),...]
    return processed_data

def crop_or_pad_final_seg(seg, shape):
    """
    This function crops or pads the segmentation to match the shape of the image
    But it is done accounting that the padding and cropping must be done in a certain specific direction 

    Args:
        seg (np.array): The segmentation to be cropped or padded
        shape (tuple): The shape of the image that the segmentation should match
    
    """

    seg_reshaped = np.zeros(shape)
    # Check if last dimension should be cropped or padded
    if (seg.shape[3] > shape[3]) and (seg.shape[1] < shape[1]) and (seg.shape[2] < shape[2]):
        seg_reshaped[-seg.shape[0]:, :seg.shape[1], :seg.shape[2], :] = seg[:, :, :, :shape[3]] 

        #seg_reshaped[-seg.shape[0]:, :seg.shape[1], :seg.shape[2], :] = seg[:, :, :, :shape[3]] 
    

    elif (seg.shape[3] > shape[3]) and (seg.shape[1] > shape[1]):
        seg_reshaped[-seg.shape[0]:, :, :seg.shape[2], :] = seg[:, :shape[1], :, :shape[3]] 
    else:
        seg_reshaped[-seg.shape[0]:, :, :seg.shape[2], :seg.shape[3]] = seg[:, :shape[1], :, :] 

    return seg_reshaped


def crop_or_pad_Bern_slices(data, new_shape):
    processed_data = np.zeros(new_shape)
    # axis 0 is the x-axis and we crop from top since aorta is at the bottom
    # axis 1 is the y-axis and we pad or crop
    # axis 2 is the z-axis and we crop from the right (end of the image) since aorta is at the left
    delta_axis0 = data.shape[0] - new_shape[0]
    if len(new_shape) == 5: # Image
        try:
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],...]
        except:
            processed_data[:, :,:,:data.shape[3],... ] = data[delta_axis0:,:new_shape[1], :new_shape[2],...]

    if len(new_shape) == 4: # Label
        try:
            processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],...]
        except:
            processed_data[:, :,:,:data.shape[3],... ] = data[delta_axis0:,:new_shape[1], :new_shape[2],...]
    return processed_data



# Predict the segmentation using model
def predict(model: torch.nn.Module, image: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probs = torch.nn.functional.softmax(logits, dim=1)
        prediction = probs.argmax(dim=1)
        return logits, probs, prediction


# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol_along_0(vol, n):    
    x = vol.shape[0]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if x > n: # original volume has more slices that the required number of slices
        vol_cropped = vol[x_s:x_s + n, :, :, :, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((n, vol.shape[1], vol.shape[2], vol.shape[3], vol.shape[4]))
        vol_cropped[x_c:x_c + x, :, :, :, :] = vol
    return vol_cropped

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol_along_1(vol, n):    
    x = vol.shape[1]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if x > n: # original volume has more slices that the required number of slices
        vol_cropped = vol[:, x_s:x_s + n, :, :, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((vol.shape[0], n, vol.shape[2], vol.shape[3], vol.shape[4]))
        vol_cropped[:, x_c:x_c + x, :, :, :] = vol
    return vol_cropped

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol_along_2(vol, n):    
    x = vol.shape[2]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if x > n: # original volume has more slices that the required number of slices
        vol_cropped = vol[:, :, x_s:x_s + n, :, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((vol.shape[0], vol.shape[1], n, vol.shape[3], vol.shape[4]))
        vol_cropped[:, :, x_c:x_c + x, :, :] = vol
    return vol_cropped

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol_along_3(vol, n):    
    x = vol.shape[3]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if x > n: # original volume has more slices that the required number of slices
        vol_cropped = vol[:, :, :, x_s:x_s + n, :]
    else: # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((vol.shape[0], vol.shape[1], vol.shape[2], n, vol.shape[4]))
        vol_cropped[:, :, :, x_c:x_c + x, :] = vol
    return vol_cropped

# ==================================================================
# crop or pad functions to change image size without changing resolution
# ==================================================================    
def crop_or_pad_4dvol(vol, target_size):
    
    vol = crop_or_pad_4dvol_along_0(vol, target_size[0])
    vol = crop_or_pad_4dvol_along_1(vol, target_size[1])
    vol = crop_or_pad_4dvol_along_2(vol, target_size[2])
    vol = crop_or_pad_4dvol_along_3(vol, target_size[3])
                
    return vol

def normalize_image_new(image):

    # ===============
    # initialize with zeros
    # ===============
    normalized_image = np.zeros((image.shape))

    # ===============
    # normalize magnitude channel
    # ===============
    normalized_image[...,0] = image[...,0] / np.amax(image[...,0])

    # ===============
    # normalize velocities
    # ===============

    # extract the velocities in the 3 directions
    velocity_image = np.array(image[...,1:4])

    # denoise the velocity vectors
    velocity_image_denoised = gaussian_filter(velocity_image, 0.5)

    # compute per-pixel velocity magnitude
    velocity_mag_image = np.linalg.norm(velocity_image_denoised, axis=-1)

    normalized_image[...,1] = 2.*(velocity_image_denoised[...,0] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
    normalized_image[...,2] = 2.*(velocity_image_denoised[...,1] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
    normalized_image[...,3] = 2.*(velocity_image_denoised[...,2] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1


    return normalized_image

def augment_data(images, # (batchsize, nx, ny, nt, 1)
                 labels, # (batchsize, nx, ny, nt)
                 data_aug_ratio,
                 gamma_min = 0.5,
                 gamma_max = 2.0,
                 brightness_min = 0.0,
                 brightness_max = 0.1,
                 noise_min = 0.0,
                 noise_max = 0.1):
        
    images_ = np.copy(images)
    labels_ = np.copy(labels)
    
    for i in range(images.shape[0]):
                        
        # ========
        # contrast # gamma contrast augmentation
        # ========
        if np.random.rand() < data_aug_ratio:
            c = np.round(np.random.uniform(gamma_min, gamma_max), 2)
            images_[i,...] = images_[i,...]**c

        # ========
        # brightness
        # ========
        if np.random.rand() < data_aug_ratio:
            c = np.round(np.random.uniform(brightness_min, brightness_max), 2)
            images_[i,...] = images_[i,...] + c
            
        # ========
        # noise
        # ========
        if np.random.rand() < data_aug_ratio:
            n = np.random.normal(noise_min, noise_max, size = images_[i,...].shape)
            images_[i,...] = images_[i,...] + n
            
    return images_, labels_