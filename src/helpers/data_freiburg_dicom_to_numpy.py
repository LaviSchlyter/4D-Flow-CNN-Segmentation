# ================================================================
# import stuff
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom as dicom
import imageio
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation')
import config.system as sys_config

# ================================================================
# Load images
# This function loads the dicom image series and converts it into a numpy array
# ================================================================
def read_dicom(pathDicom):
    
    # ================================
    # create a list of all dicom slices filenames
    # ================================
    list_of_dicom_filenames = []
    for dirName, subdirList, fileList in os.walk(pathDicom):
        fileList.sort()
        for filename in fileList:
            list_of_dicom_filenames.append(os.path.join(dirName,filename))
   
    # ================================
    # find the number of slice locations and number of time steps
    # ================================
    instance_nums = []
    slice_locations = []
    acquisition_times = []
    
    for i in range(len(list_of_dicom_filenames)):
        ds = dicom.read_file(list_of_dicom_filenames[i])    
        instance_nums.append(ds.InstanceNumber)
        slice_locations.append(ds.SliceLocation)
        acquisition_times.append(ds.AcquisitionTime)

    # ================================
    # convert lists to arrays
    # ================================    
    instance_nums = np.array(instance_nums)
    slice_locations = np.array(slice_locations)
    acquisition_times = np.array(acquisition_times)
    
    # ================================
    # each slice location is acquired for each acquisition time - thus, there are several slices with the same slice location
    # similiarly, at each acquisition time all slices are acquried - thus, there are several slices with the same acquisition time
    # Thus, we take the unique slice locations and acquisition times to find the number of slice locations and time steps
    # ================================
    unique_slice_locations = np.sort(np.unique(slice_locations))
    unique_acquisition_times = np.sort(np.unique(acquisition_times))
    
    num_slices_per_volume = unique_slice_locations.shape[0]
    num_times_in_acquisition = unique_acquisition_times.shape[0]

    # ================================
    # get a reference 2d slice and extract some info from the file header
    # ================================
    refDs = dicom.read_file(list_of_dicom_filenames[0])
    PixelDims = (int(refDs.Rows), int(refDs.Columns), num_slices_per_volume, num_times_in_acquisition)
    PixelSpacing = (float(refDs.PixelSpacing[0]), float(refDs.PixelSpacing[1]), float(refDs.SliceThickness))
    print('Details from the image header:')
    print('Image dimensions: ', PixelDims)
    print('Pixel spacing in mm: ', PixelSpacing)

    # ================================
    # initialize a zeros array of the size of the image
    # ================================
    img = np.zeros(PixelDims, dtype = refDs.pixel_array.dtype)

    # ================================
    # loop through all the DICOM files and populate the array
    # ================================
    for i in range(len(list_of_dicom_filenames)):
        ds = dicom.read_file(list_of_dicom_filenames[i])    
        slice_idx = np.where(ds.SliceLocation == unique_slice_locations)[0][0]
        time_idx = np.where(ds.AcquisitionTime == unique_acquisition_times)[0][0]                             
        img[:, :, slice_idx, time_idx] = ds.pixel_array

    return img

# ================================================================
# This function takes a 4d image array [imgx, imgy, imgz, time] and saves a gif of all xy slices over all times.
# ================================================================
def save_gif(im, save_path):

    num_times = im.shape[3]
    num_slices = im.shape[2]

    # ================================
    # all slices of each time-index as png
    # ================================
    for t in range(num_times):    
        plt.figure(figsize=[30,30])
        for j in range(num_slices):
            plt.subplot(6, 6, j+1)
            plt.imshow(im[:,:,j,t], cmap='gray')
            plt.xticks([], []); plt.yticks([], [])
            plt.title('M, s: ' + str(j) + ', t:' + str(t//2))
        plt.savefig(save_path + '_time' + str(t) + '.png')
        plt.close()

    # ================================
    # all time indices in gif
    # ================================
    gif = []
    for t in range(num_times):
        gif.append(imageio.imread(save_path + '_time' + str(t) + '.png'))
        
    # ================================
    # save gif
    # ================================
    imageio.mimsave(save_path + '.gif', gif, format='GIF', duration=0.5)
    
# ================================
# This function deletes all pngs in the folder
# ================================
def remove_pngs(dir_name):
    list_of_files = os.listdir(dir_name)
    for item in list_of_files:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_name, item))
    
# ====================
# set basepaths for reading and writing
# ====================
basepath_reading = sys_config.orig_data_root #aka '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/freiburg/main_data_transfer/126_subjects'
basepath_writing = sys_config.project_data_freiburg_root 

# ====================
# walk through all the subjects, read the dicoms and save as npy arrays
# ====================
# for each subject, the directory structure is something like this:
# <subject_id (1-3 digit no.)>/   ----> level0
# <8-char-dir-name>/   ----> level1
# <8-char-dir-name>/   ----> level2
# <two 8-char-dirs containing phase and magnitude>   ----> level3
# Actually, in level 3, there are potentially more than 2 directories, but we want to choose the ones with the two largest number of files.
# ====================
_, dir_names_level0, _ = next(os.walk(basepath_reading))

# ====================        
# for each subject
# ====================        
for dir_name_this_subject_level0 in dir_names_level0:

    dir_path_this_subject_level0 = os.path.join(basepath_reading, dir_name_this_subject_level0)
    print('========================================')
    print('Path level 0: ' + dir_path_this_subject_level0) # e.g. basepath/10

    # ====================        
    # walk through the directories for this subject - level1
    # ====================
    _, dir_names_level1, _ = next(os.walk(dir_path_this_subject_level0))
    for dir_name_this_subject_level1 in dir_names_level1:
        # for some subjects, there are two directories at level1. They are seemingly just copies of each other. Saving both nonetheless.
        dir_path_this_subject_level1 = os.path.join(dir_path_this_subject_level0, dir_name_this_subject_level1)
        print('Path level 1: ' + dir_path_this_subject_level1) # e.g. basepath/10/100196E0

        # ====================        
        # walk through the directories for this subject - level2
        # ====================                
        _, dir_names_level2, _ = next(os.walk(dir_path_this_subject_level1))
        for dir_name_this_subject_level2 in dir_names_level2:
            dir_path_this_subject_level2 = os.path.join(dir_path_this_subject_level1, dir_name_this_subject_level2)
            print('Path level 2: ' + dir_path_this_subject_level2) # e.g. basepath/10/100196E0/100196E1
            
            # Define the save directory
            save_dir = os.path.join(basepath_writing, dir_name_this_subject_level0, dir_name_this_subject_level1, dir_name_this_subject_level2)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'image.npy')

            # Check if 'image.npy' already exists, skip this subject if it does
            if os.path.exists(save_path):
                print(f'Subject {dir_name_this_subject_level0} at path {save_path} already processed. Skipping...')
                continue  # Skip this subject as it's already processed

            # ====================            
            # walk through the directories for this subject - level3
            # ====================
            _, dir_names_level3, _ = next(os.walk(dir_path_this_subject_level2))
            
            # ====================            
            # At this level, the directories will contain .dcm files.
            # Of the possibly many directories at this level, we want to choose the ones with the two largest number of files
            # ====================
            dir_sizes = []
            for dir_name_this_subject_level3 in dir_names_level3:
                dir_path_this_subject_level3 = os.path.join(dir_path_this_subject_level2, dir_name_this_subject_level3)
                print('Path level 3: ' + dir_path_this_subject_level3)
                dir_sizes.append(os.path.getsize(dir_path_this_subject_level3))                                
            
            # ====================
            # sort dirs by size
            # ====================
            dir_sizes_sorted = sorted(dir_sizes, reverse=True)                            
            index = [dir_sizes_sorted.index(x) for x in dir_sizes]
            # the directory with the highest number of files is the phase directory
            phase_dir = os.path.join(dir_path_this_subject_level2, dir_names_level3[index.index(0)])
            # the directory with the 2nd highest number of files is the magnitude directory
            # (contains 1/3rd the number of files in the phase directory)
            magnitude_dir = os.path.join(dir_path_this_subject_level2, dir_names_level3[index.index(1)])
                                                        
            # ================================
            # read the dicom magnitude and phase image series and convert them to numpy arrays
            # ================================
            print('====================')
            print('Reading magnitude dicom series from: ' + magnitude_dir)
            image_mag = read_dicom(magnitude_dir)
            print('Reading phase dicom series from: ' + phase_dir)
            image_pha = read_dicom(phase_dir)
            image_phx = image_pha[:,:,:,0::3]
            image_phy = image_pha[:,:,:,1::3]
            image_phz = image_pha[:,:,:,2::3]
            image = np.stack((image_mag, image_phx, image_phy, image_phz),axis=-1)
        
            # ================================
            # save the 4d flow image as a npy array
            # ================================
            
            
            print('# Saving combined magnitude and phase image ([imgx, imgy, imgz, time, 4]) as a numpy array...')
            np.save(save_path, image)
            print('Image saved at: ', save_path)
            print('Image shape: ', image.shape)
            print('====================')
        
            # ================================
            # save the 4d flow image as gifs
            # ================================
            save_gif_bool = False # saving the gif is nice for visualization, but it takes about 5 minutes for each image.
            if save_gif_bool is True:
                im = np.load(save_path)
                print('# ======== Creating gif for the magnitude images...')
                save_gif(im[:,:,:,:,0], save_dir + 'mag')
                print('# ======== Creating gif for the phase x images...')
                save_gif(im[:,:,:,:,1], save_dir + 'phx')
                print('# ======== Creating gif for the phase y images...')
                save_gif(im[:,:,:,:,2], save_dir + 'phy')
                print('# ======== Creating gif for the phase z images...')
                save_gif(im[:,:,:,:,3], save_dir + 'phz')
                                            
                # ================================
                # delete all pngs created by the previous step
                # ================================
                print('# ======== Cleaning up...')
                remove_pngs(save_dir)
                
print('========================================')
print('Done.')
print('========================================')
