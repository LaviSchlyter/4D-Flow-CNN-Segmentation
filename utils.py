import os
import glob
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import morphology
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import torch
    
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
        processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,...]
    return processed_data



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

