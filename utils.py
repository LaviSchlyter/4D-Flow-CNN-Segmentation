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
    
    # velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
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
    # axis 1 is the y-axis and we pad
    # axis 2 is the z-axis and we crop from the right (end of the image) since aorta is at the left
    delta_axis0 = data.shape[0] - new_shape[0]
    if len(new_shape) == 5: # Image
        # The x is always cropped, y always padded, z_cropped
        processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],...]

    if len(new_shape) == 4: # Label
        # The x is always cropped, y always padded, z_cropped
        processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],...]
    return processed_data



# Predict the segmentation using model
def predict(model: torch.nn.Module, image: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probs = torch.nn.functional.softmax(logits, dim=1)
        prediction = probs.argmax(dim=1)
        return logits, probs, prediction




# ==========================================================
# ==========================================================
# NOT USED  BELOW
# ==========================================================
# ==========================================================
































# ==========================================================
# ==========================================================       
def save_sample_image_and_labels_across_z(x, # nz, nx, ny, nt, 4
                                          y, # nz, nx, ny, nt
                                          savepath):
    
    nz = x.shape[0]
    nt = x.shape[3]
    ids_z = np.arange(0, nz, nz//8)
    ids_t = np.arange(0, nt, nt//8)
    nc = len(ids_z)
    nr = len(ids_t)
            
    for r in range(nr):

        plt.figure(figsize=[3*nc, 3*nr])
            
        for c in range(nc): 
            
            x_c1 = x[ids_z[c], :, :, ids_t[r], 0]
            x_c2 = x[ids_z[c], :, :, ids_t[r], 1]
            x_c3 = x[ids_z[c], :, :, ids_t[r], 2]
            x_c4 = x[ids_z[c], :, :, ids_t[r], 3]
            y_ = y[ids_z[c], :, :, ids_t[r]]
            
            title = 'z_' + str(ids_z[c]) + '_'
            
            plt.subplot(nr, nc, nc*0 + c + 1); plt.imshow(x_c1, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c1')
            plt.subplot(nr, nc, nc*1 + c + 1); plt.imshow(x_c2, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c2')
            plt.subplot(nr, nc, nc*2 + c + 1); plt.imshow(x_c3, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c3')
            plt.subplot(nr, nc, nc*3 + c + 1); plt.imshow(x_c4, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c4')
            plt.subplot(nr, nc, nc*4 + c + 1); plt.imshow(y_, cmap='tab20'); plt.colorbar(); plt.title(title + 'gt')
            
        plt.savefig(savepath + '_t' + str(ids_t[r]) + '.png', bbox_inches='tight')
        plt.close()
        
# ==========================================================
# ==========================================================       
def save_sample_image_and_labels_across_t(x, # nz, nx, ny, nt, 4
                                          y, # nz, nx, ny, nt
                                          savepath):
    
    nz = x.shape[0]
    nt = x.shape[3]
    
    ids_z = np.arange(0, nz, nz//8)
    ids_t = np.arange(0, nt, nt//8)
    
    nc = len(ids_t)
    nr = len(ids_z)
            
    for r in range(nr):

        plt.figure(figsize=[3*nc, 3*nr])
            
        for c in range(nc): 
            
            x_c1 = x[ids_z[r], :, :, ids_t[c], 0]
            x_c2 = x[ids_z[r], :, :, ids_t[c], 1]
            x_c3 = x[ids_z[r], :, :, ids_t[c], 2]
            x_c4 = x[ids_z[r], :, :, ids_t[c], 3]
            y_ = y[ids_z[r], :, :, ids_t[c]]
            
            title = 't_' + str(ids_t[c]) + '_'
            
            plt.subplot(nr, nc, nc*0 + c + 1); plt.imshow(x_c1, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c1')
            plt.subplot(nr, nc, nc*1 + c + 1); plt.imshow(x_c2, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c2')
            plt.subplot(nr, nc, nc*2 + c + 1); plt.imshow(x_c3, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c3')
            plt.subplot(nr, nc, nc*3 + c + 1); plt.imshow(x_c4, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c4')
            plt.subplot(nr, nc, nc*4 + c + 1); plt.imshow(y_, cmap='tab20'); plt.colorbar(); plt.title(title + 'gt')
            
        plt.savefig(savepath + '_z' + str(ids_z[r]) + '.png', bbox_inches='tight')
        plt.close()
        
# ==========================================================
# ==========================================================       
def save_sample_results(im, # nz, nx, ny, nt, 4
                        pr, # nz, nx, ny, nt
                        gt, # nz, nx, ny, nt
                        filename):
    
    nz = im.shape[0]
    nt = im.shape[3]
    ids_z = np.arange(0, nz, nz//8)
    ids_t = np.arange(0, nt, nt//8)
    nc = len(ids_z)
    nt = len(ids_t)
            
    for t in range(nt):

        nr = 6
        plt.figure(figsize=[3*nc, 3*nr])
            
        for c in range(nc): 
            
            x_c1 = im[ids_z[c], :, :, ids_t[t], 0]
            x_c2 = im[ids_z[c], :, :, ids_t[t], 1]
            x_c3 = im[ids_z[c], :, :, ids_t[t], 2]
            x_c4 = im[ids_z[c], :, :, ids_t[t], 3]
            pr_ = pr[ids_z[c], :, :, ids_t[t]]
            gt_ = gt[ids_z[c], :, :, ids_t[t]]
            
            title = 'z_' + str(ids_z[c]) + '_'
            
            plt.subplot(nr, nc, nc*0 + c + 1); plt.imshow(x_c1, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c1')
            plt.subplot(nr, nc, nc*1 + c + 1); plt.imshow(x_c2, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c2')
            plt.subplot(nr, nc, nc*2 + c + 1); plt.imshow(x_c3, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c3')
            plt.subplot(nr, nc, nc*3 + c + 1); plt.imshow(x_c4, cmap='gray'); plt.colorbar(); plt.title(title + 'x_c4')
            plt.subplot(nr, nc, nc*4 + c + 1); plt.imshow(pr_, cmap='tab20'); plt.colorbar(); plt.title(title + 'pred')
            plt.subplot(nr, nc, nc*5 + c + 1); plt.imshow(gt_, cmap='tab20'); plt.colorbar(); plt.title(title + 'GT')
            
        plt.savefig(filename + '_t' + str(ids_t[t]) + '.png', bbox_inches='tight')
        plt.close()
        
# ================================================================== 
# Computes hausdorff distance between binary labels (compute separately for each label)
# ==================================================================    
def compute_surface_distance_per_label(y_1,
                                       y_2,
                                       sampling = 1,
                                       connectivity = 1):

    y1 = np.atleast_1d(y_1.astype(np.bool))
    y2 = np.atleast_1d(y_2.astype(np.bool))
    
    conn = morphology.generate_binary_structure(y1.ndim, connectivity)

    S1 = y1.astype(np.float32) - morphology.binary_erosion(y1, conn).astype(np.float32)
    S2 = y2.astype(np.float32) - morphology.binary_erosion(y2, conn).astype(np.float32)
    
    S1 = S1.astype(np.bool)
    S2 = S2.astype(np.bool)
    
    dta = morphology.distance_transform_edt(~S1, sampling)
    dtb = morphology.distance_transform_edt(~S2, sampling)
    
    sds = np.concatenate([np.ravel(dta[S2 != 0]), np.ravel(dtb[S1 != 0])])
    
    return sds

# ==================================================================   
# ==================================================================   
def compute_surface_distance(y1,
                             y2,
                             nlabels):
    
    mean_surface_distance_list = []
    hausdorff_distance_list = []
    
    for l in range(1, nlabels):

        surface_distance = compute_surface_distance_per_label(y_1 = (y1 == l),
                                                              y_2 = (y2 == l))
    
        mean_surface_distance = surface_distance.mean()
        # hausdorff_distance = surface_distance.max()
        hausdorff_distance = np.percentile(surface_distance, 95)

        mean_surface_distance_list.append(mean_surface_distance)
        hausdorff_distance_list.append(hausdorff_distance)
        
    return np.array(hausdorff_distance_list)

# ===========================      
# data augmentation: gamma contrast, brightness (one number added to the entire slice), additive noise (random gaussian noise image added to the slice)
# ===========================        
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