# ==========================================
# import stuff
# ==========================================
import os
import h5py
import numpy as np
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation')
import config.system as sys_config
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/src')
from helpers import data_freiburg_subject_ordering as subjects_ordering
from helpers import utils

# ==========================================
# ==========================================
def crop_or_pad_zeros(data, new_shape):
        
    processed_data = np.zeros(new_shape)

    # ======================
    # axes 0 (x) and 1 (y) need to be cropped
    # ======================
    # axis 0 will be cropped only from the top (as the aorta often exists until the bottom end of the image)
    # axis 1 will be cropped evenly from the right and left
    # ======================
    # axes 2 (z) and 3 (t) will be padded with zeros if the original image is smaller
    # ======================
    
    delta_axis0 = data.shape[0] - new_shape[0]
    delta_axis1 = data.shape[1] - new_shape[1]
        
    if len(new_shape) is 5: # image
        processed_data[:, :, :data.shape[2], :data.shape[3], :] = data[delta_axis0:, (delta_axis1//2):-(delta_axis1//2), :, :, :]
    elif len(new_shape) is 4: # label
        processed_data[:, :, :data.shape[2], :data.shape[3]] = data[delta_axis0:, (delta_axis1//2):-(delta_axis1//2), :, :]
    
    return processed_data
            
# ==========================================
# Read all images and the figure out the min and max dimensions along each dimension
# ==========================================
def find_shapes(num_images = len(subjects_ordering.SUBJECT_DIRS)):
        
    images_shapes = []
    for n in range(num_images):
    
        if n%10 is 0:
            print('Reading image ' + str(n) + ' out of ' + str(num_images) + '...')
        
        images_shapes.append(np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy').shape)
    
    images_shapes = np.array(images_shapes)
    
    # ================
    # prining stats over all subjects, we get:
    # ================
    print('=== x'); print(np.min(images_shapes[:,0])); print(np.median(images_shapes[:,0])); print(np.max(images_shapes[:,0]))
    print('=== y'); print(np.min(images_shapes[:,1])); print(np.median(images_shapes[:,1])); print(np.max(images_shapes[:,1]))
    print('=== z'); print(np.min(images_shapes[:,2])); print(np.median(images_shapes[:,2])); print(np.max(images_shapes[:,2]))
    print('=== t'); print(np.min(images_shapes[:,3])); print(np.median(images_shapes[:,3])); print(np.max(images_shapes[:,3]))
    # Min-Median-Max-Dimension
    # 160-160-160-0 (x)
    # 120-120-150-1 (y)
    # 29-30-32-2 (z)
    # 21-28-57-3 (t)
    # ================
    # prining stats over the first 29 subjects (that have been segmented by Nicolas), we get:
    # ================
    # Min-Median-Max-Dimension
    # 160-160-160-0 (x)
    # 120-120-150-1 (y)
    # 30-30-32-2 (z)
    # 23-40-48-3 (t)
    
    return images_shapes

# ==========================================
# ==========================================       
def prepare_and_write_data(basepath,
                           filepath_output,
                           idx_start,
                           idx_end,
                           train_test):
        
    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'), 
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [144, 112, 32, 48, 4] # [x, y, z, t, num_channels]
    common_label_shape = [144, 112, 32, 48] # [x, y, z, t]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)
    
    # ==========================================
    # ==========================================
    num_images_to_load = idx_end + 1 - idx_start
    
    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [common_image_shape[2]*num_images_to_load,
                            common_image_shape[0],
                            common_image_shape[1],
                            common_image_shape[3],
                            common_image_shape[4]]
    
    labels_dataset_shape = [common_label_shape[2]*num_images_to_load,
                            common_label_shape[0],
                            common_label_shape[1],
                            common_label_shape[3]]  
        
    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w") 
    
    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================    
    dataset['images_%s' % train_test] = hdf5_file.create_dataset("images_%s" % train_test, images_dataset_shape, dtype='float32')       
    dataset['labels_%s' % train_test] = hdf5_file.create_dataset("labels_%s" % train_test, labels_dataset_shape, dtype='uint8')       
           
    i = 0
    for n in range(idx_start, idx_end + 1): 
        
        print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        
        # load the numpy image (saved by the dicom2numpy file)
        image_data = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy')                  
        # normalize the image
        image_data = utils.normalize_image(image_data)
        # make all images of the same shape
        image_data = crop_or_pad_zeros(image_data, common_image_shape)                  
        # move the z-axis to the front, as we want to concantenate data along this axis
        image_data = np.moveaxis(image_data, 2, 0)                         
        # add the image to the hdf5 file
        dataset['images_%s' % train_test][i*common_image_shape[2]:(i+1)*common_image_shape[2], :, :, :, :] = image_data
    
        # load the numpy label (saved by the random walker segmenter)
        label_data = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/random_walker_prediction.npy')                  
        # make all images of the same shape
        label_data = crop_or_pad_zeros(label_data, common_label_shape)                  
        # move the z-axis to the front, as we want to concantenate data along this axis
        label_data = np.moveaxis(label_data, 2, 0)  
        # cast labels as uints
        label_data = label_data.astype(np.uint8)                       
        # add the image to the hdf5 file
        dataset['labels_%s' % train_test][i*common_label_shape[2]:(i+1)*common_label_shape[2], :, :, :] = label_data
        
        # increment the index being used to write in the hdf5 datasets
        i = i + 1
    
    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0
       
# ==========================================
# ==========================================       
def load_data(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False):
    
    # ==========================================
    # define file paths for images and labels 
    # ==========================================
    dataset_filepath = basepath + '/images_and_labels_from' + str(idx_start) + 'to' + str(idx_end) + '.hdf5'
    
    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_data(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                               idx_end = idx_end,
                               train_test = train_test)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')    
    
# ===============================================================
# Main function that runs if this file is run directly
# ===============================================================
if __name__ == '__main__':

    # ==========================================
    # The original dicom images have been saved as numpy arrays at this basepath
    # ==========================================
    basepath = sys_config.project_data_freiburg_root # '/usr/bmicnas01/data-biwi-01/nkarani/students/nicolas/data/freiburg'

    data_train = load_data(basepath = basepath, idx_start = 0, idx_end = 19, train_test='train')
    data_val = load_data(basepath = basepath, idx_start = 20, idx_end = 24, train_test='validation')    
    data_test = load_data(basepath = basepath, idx_start = 25, idx_end = 28, train_test='test')
    
    
# ===============================================================
# End of file
# ===============================================================