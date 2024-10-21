"""
Here we will take the cross sectional slices to the aorta by following the procedure in
'/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4DFlowAorticAnomalies/src/helpers/data_bern_numpy_to_preprocessed_hdf5.py'
"""
import os
import sys
import h5py
import numpy as np
import logging
import argparse
from matplotlib import pyplot as plt
import yaml
import re
from skimage.morphology import skeletonize_3d, dilation, cube, binary_erosion


# Define base paths for your projects
basepath = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/"
# Append the path for CNN-segmentation
cnn_segmentation_path = os.path.join(basepath, 'CNN-segmentation/')
sys.path.insert(0, cnn_segmentation_path)
from config import system as sys_config
from src.helpers.utils import gather_experiment_paths
sys.path.remove(cnn_segmentation_path)
if 'src.helpers.utils' in sys.modules:
    del sys.modules['src.helpers.utils']

four_d_flow_path = os.path.join(basepath, '4DFlowAorticAnomalies/')
sys.path.insert(0, four_d_flow_path)
from src.helpers.utils import order_points, skeleton_points, make_dir_safely, normalize_image, interpolate_and_slice, crop_or_pad_normal_slices
sys.path.remove(four_d_flow_path)
if 'src.helpers.utils' in sys.modules:
    del sys.modules['src.helpers.utils']

def save_images_cropped_sliced_masked(data_path, folder_save, h5_path, patient_type = "controls", full_aorta=False, suffix = ''):
    n_images = 64
    grid = 8
    size = 14
    aorta = ''
    if full_aorta:
        n_images = 256
        grid = 16
        size = 20
        aorta = '_full_aorta'
        
    #if hand_seg:
    #    experiment_path = os.path.join('/',*data_path.split('/')[:-2])
    #    folder_save = os.path.join(experiment_path, 'HandSegmentedCenterlinePostProcessing')
    #else:
    #    experiment_path = os.path.join('/',*data_path.split('/')[:-1])
    #    folder_save = os.path.join(experiment_path, 'CenterlinePostProcessingViz')

    h5_file_path = os.path.join(h5_path, f'{patient_type}{suffix}_masked_sliced_images.hdf5')
    sliced_data = h5py.File(h5_file_path, 'r')
    images_cropped_sliced = sliced_data[f'sliced_images_{patient_type}']
    # Create the folder to save the images
    make_dir_safely(folder_save + f'/masked_cropped_sliced{aorta+suffix}')
    data_list = os.listdir(data_path)
    data_list.sort()

    for n, p in enumerate(data_list):
        
        patient = re.split(r'seg_|.npy', p)[1]
        fig, axs = plt.subplots(grid,grid, figsize=(size,size))
        ax = axs.ravel()
        for i in range(n_images):
            ax[i].imshow(images_cropped_sliced[i+(n*n_images), :,:, 3,1])
            ax[i].axis('off')
        plt.savefig(os.path.join(folder_save + f'/masked_cropped_sliced{aorta+suffix}', f'{patient}_masked_sliced{aorta}.png'))
        plt.close()
    sliced_data.close()
    logging.info(f'Images saved in {folder_save + f"/masked_cropped_sliced{aorta+suffix}"}')
    

    



def prepare_and_write_masked_data_sliced_bern(model_path, filepath_output, patient_type, cnn_predictions = True, suffix =''):

    # Looking at all the shapes for z slices and time we get the following
    common_image_shape = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]
    end_shape = [32, 32, 64, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    img_path = f'{sys_config.project_data_root}/preprocessed/{patient_type}/numpy{suffix}'
    seg_path = model_path
    patients = os.listdir(seg_path)
    # Sort the patients
    patients.sort()
    num_images_to_load = len(patients)

    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [end_shape[2]*num_images_to_load,
                            end_shape[0],
                            end_shape[1],
                            end_shape[3],
                            end_shape[4]]

    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")
    dataset['sliced_images_%s' % patient_type] = hdf5_file.create_dataset("sliced_images_%s" % patient_type, images_dataset_shape, dtype='float32')

    i=0
    for patient in patients:
        # log progress and patient
        logging.info('Processing patient number %s of %s' % (i+1, num_images_to_load))
        logging.info('Processing patient %s' % patient)

        #Load the image and segmentation
        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        segmented_original = np.load(os.path.join(seg_path, patient))

        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        segmented = dilation(segmented_original[:,:,:,3], cube(3))
        temp_for_stack = [segmented for i in range(time_steps)]
        segmented = np.stack(temp_for_stack, axis=3)
        image = image.astype(float)

        image = normalize_image(image)
        temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        # Extract the centerline points using skeletonization
        points = skeleton_points(segmented_original, dilation_k = 4, erosion_k = 4)
        try:

            try:
                points_order_ascending_aorta = order_points(points[::-1], angle_threshold=3/2*np.pi/2.)
                logging.info('points order ascending aorta with angle threshold 3/2*np.pi/2.')
            except Exception as e:
                try:
                    points_order_ascending_aorta = order_points(points[::-1], angle_threshold=1/2*np.pi/2.)
                    logging.info('points order ascending aorta with angle threshold 1/2*np.pi/2.')
                except Exception as e:
                    points_order_ascending_aorta = np.array([0,0,0])
                    logging.info(f'An error occurred while processing {patient} ascending aorta: {e}')
            
            points = points[(points[:, 0] <= 90) & (points[:, 0] >= points_order_ascending_aorta[0][0]) & (points[:, 1] <= points_order_ascending_aorta[0][1])]

            temp = []        
            for index, element in enumerate(points[2:]):
                if (index%2)==0:
                    temp.append(element)
            coords = np.array(temp)
            #===========================================================================================
            # Parameters for the interpolation and creation of the files
            # We create Slices across time and channels in a double for loop
            temp_for_channel_stacking = []
            for channel in range(image.shape[4]):
                temp_for_time_stacking = []
                for time in range(image.shape[3]):
                    slice_dict = interpolate_and_slice(image[:,:,:,time,channel], coords, common_image_shape)
                    straightened = slice_dict['straightened']
                    temp_for_time_stacking.append(straightened)
                channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
                temp_for_channel_stacking.append(channel_stacked)
            straightened = np.stack(temp_for_channel_stacking, axis=-1)
            image_out = straightened

            # make all images of the same shape
            logging.info(f'Image shape before cropping and padding: {image_out.shape}')
            image_out = crop_or_pad_normal_slices(image_out, end_shape)
            logging.info(f'Image shape after cropping and padding: {image_out.shape}')

            # move the z-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 2, 0)
            
            # add the image to the hdf5 file
            dataset['sliced_images_%s' % patient_type][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

            i = i + 1
        except Exception as e:
            logging.info(f'An error occurred while processing {patient}: {e}')
            i = i + 1
    
    hdf5_file.close()
    return 0


if __name__ == "__main__":
    result_dir = os.path.join(sys_config.project_code_root, 'Results')
                
    default_models = ['231116-1134_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_only_w_bern']
    class_labels = ['controls', 'patients', 'patients_compressed_sensing', 'controls_compressed_sensing'] # ['controls', 'patients', 'patients_compressed_sensing', 'controls_compressed_sensing']
    parser = argparse.ArgumentParser(description="Run inference with specific settings.")
    parser.add_argument('--config_path', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Check if configuration file is provided else use the default experiment list
    if args.config_path:
        with open(args.config_path, 'r') as file:
            config = yaml.safe_load(file)
            experiment_names_path = gather_experiment_paths(result_dir, config['filters']['specific_times'])
            models = [os.path.basename(exp) for exp in experiment_names_path]
            class_labels = config['class_labels']

    else:
        models = default_models
    
    for model in models:
        logging.info(f'Processing model: {model}')
        for patient_type in class_labels:
            logging.info(f'Processing patient type: {patient_type}')
            if patient_type.__contains__('compressed_sensing'):
                compressed_sensing_data = True
                suffix = '_compressed_sensing'
                patient_type = patient_type.replace('_compressed_sensing', '')
            else:
                compressed_sensing_data = False
                suffix = ''

            data_path = os.path.join(result_dir, model, patient_type+suffix)
            experiment_path = os.path.join('/',*data_path.split('/')[:-1])
            filepath_output = os.path.join(experiment_path, f'{patient_type+suffix}_masked_sliced_images.hdf5')
            if not os.path.exists(filepath_output):
                logging.info(f'Writing file: {filepath_output}')
                masked_sliced_data = prepare_and_write_masked_data_sliced_bern(data_path, filepath_output, patient_type, cnn_predictions = True, suffix = suffix)
            else:
                logging.info(f'File already exists: {filepath_output}')
            
            folder_save = os.path.join(experiment_path, 'CenterlinePostProcessingViz')
            save_images_cropped_sliced_masked(data_path, folder_save, experiment_path, patient_type, full_aorta=False, suffix = suffix)



            




                                                