import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import argparse
import yaml


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
# Append the path for 4DFlowAorticAnomalies, ensuring it is at the beginning of the path list
four_d_flow_path = os.path.join(basepath, '4DFlowAorticAnomalies/')
sys.path.insert(0, four_d_flow_path)
from src.helpers.utils import order_points, skeleton_points, make_dir_safely
sys.path.remove(four_d_flow_path)
if 'src.helpers.utils' in sys.modules:
    del sys.modules['src.helpers.utils']

def save_all_centerlines_in_one_image(folder_save, suffix =''):
    save_images_path = os.path.join(folder_save, f'SubjectCenterlines{suffix}')
    image_list = os.listdir(save_images_path)
    image_list.sort()
    n_images = len(image_list)
    logging.info(f"Saving {n_images} images in one image")
    fig, ax = plt.subplots(np.ceil(np.sqrt(n_images)).astype(int), np.ceil(np.sqrt(n_images)).astype(int), figsize=(20,20))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, file in enumerate(image_list):
        img = plt.imread(os.path.join(save_images_path, file))
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].imshow(img)
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].set_title(file)
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].axis('off')

    plt.savefig(os.path.join(folder_save, f'SubjectCenterlines{suffix}.png'))
    plt.close(fig)
    logging.info(f"Saved images to {os.path.join(folder_save, f'SubjectCenterlines{suffix}.png')}")


def plot_side_by_side(img, points, modified_points, folder_save, subfolder, name, z_slice = 15, time_step = 3, channel = 0, interpolation = False):
    fig, ax = plt.subplots(1, 2, figsize=(7, 7))
    axs = ax.ravel()
    axs[0].imshow(img[:, :, z_slice, time_step, channel], cmap='gray')
    axs[0].scatter(points[:, 1], points[:, 0], c='red', s=2, marker='o')

    axs[1].imshow(img[:, :, z_slice, time_step, channel], cmap='gray')
    if interpolation:
        axs[1].scatter(modified_points[:, 2], modified_points[:, 1], c='red', s=2, marker='o')
    else:
        axs[1].scatter(modified_points[:, 1], modified_points[:, 0], c='red', s=2, marker='o')

    name_save = os.path.join(folder_save, subfolder, f'{name}.png')
    logging.info(f"Saving {subfolder} visualization to {name_save}")
    fig.savefig(name_save)
    plt.close(fig)
def spline_interpolation(coords, n_points = 64, smooth = 10):
    x = coords[:, 1]
    y = coords[:, 0]
    z = coords[:, 2]
    coords = np.array([z, y, x]).transpose([1, 0])
    params = [i/ (n_points-1) for i in range(n_points)]
    tck, _ = interpolate.splprep(np.swapaxes(coords, 0, 1), s=smooth, k = 3)
    new_points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)
    return new_points

def create_centerlines(data_path, folder_save, patient_type = "controls", compressed_sensing = False, cnn_predictions = True, z_slice = 15, time_step = 3, channel = 0):
    
    suffix = "_compressed_sensing" if compressed_sensing else ""
    img_path = f'{sys_config.project_data_root}/preprocessed/{patient_type}/numpy{suffix}'

    logging.info(f"Loading images from {img_path}")
    
    centerline_folder = 'SubjectCenterlines' + suffix
    interpolation_folder = 'SubjectCenterlines_and_interpolation' + suffix
    full_aorta_folder = 'SubjectCenterlines_and_interpolation_full_aorta' + suffix
    #if hand_segmented_path:
    #    experiment_path = os.path.join('/',*data_path.split('/')[:-2])    
    #    folder_save = os.path.join(experiment_path, 'HandSegmentedCenterlinePostProcessing')
    #else:
    #    experiment_path = os.path.join('/',*data_path.split('/')[:-1])
    #    folder_save = os.path.join(experiment_path, 'CenterlinePostProcessingViz')

    
    for folder in [centerline_folder, interpolation_folder, full_aorta_folder]:
        make_dir_safely(os.path.join(folder_save, folder))
    

    for n, patient in enumerate(os.listdir(data_path)):
        logging.info(f"Processing patient {n+1}/{len(os.listdir(data_path))}")
        
        # Since some of the data does not have _.npy but rather .npy we need to take care of that
        name = patient.replace("seg_", "").replace("_.npy", "").replace(".npy", "")
        # Check if the patient has already been processed
        if os.path.exists(os.path.join(folder_save, centerline_folder, f'{name}.png')):
            logging.info(f"Patient {name} has already been processed. Skipping.")
            continue
        logging.info(f"Processing patient {name}")

        # Load the image
        img = np.load(os.path.join(img_path, f'{patient.replace("seg_", "")}'))
        # Load the segmentation
        seg = np.load(os.path.join(data_path, patient))

        # Morphological operations to remove noise and fill holes
        # Because cnn prediction we use those kernels
        points_not_dilated = skeleton_points(seg, dilation_k = 0)
        points = skeleton_points(seg, dilation_k = 4,erosion_k = 4)

        # ==================================================================
        # Visualization of the centerline
        # ==================================================================
        
        plot_side_by_side(img, points_not_dilated, points, folder_save, centerline_folder, name, z_slice = z_slice, time_step = time_step, channel = channel, interpolation = False)
        # ==================================================================
        # Limit to ascending aorta and interpolate
        # ==================================================================

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
        
        points_ao = points[(points[:, 0] <= 90) & (points[:, 0] >= points_order_ascending_aorta[0][0]) & (points[:, 1] <= points_order_ascending_aorta[0][1])]

        temp_ao = []
        for index, element in enumerate(points_ao[2:]):
            if (index%2==0):
                temp_ao.append(element)
        coords_ao = np.array(temp_ao)

        new_points_ao = spline_interpolation(coords_ao, n_points = 64, smooth = 10)
        plot_side_by_side(img, points_ao, new_points_ao, folder_save, interpolation_folder, name, z_slice = z_slice, time_step = time_step, channel = channel, interpolation = True)

        # ==================================================================
        # Full aorta if possible
        # ==================================================================

        try:
            points_order_full_aorta = order_points(points)
            temp_full_aorta = []
            for index, element in enumerate(points_order_full_aorta[2:]):
                if (index%2==0):
                    temp_full_aorta.append(element)
            coords_full_aorta = np.array(temp_full_aorta)
            new_points_full_aorta = spline_interpolation(coords_full_aorta, n_points = 256, smooth = 10)
            plot_side_by_side(img, points_order_full_aorta, new_points_full_aorta, folder_save, full_aorta_folder, name, z_slice = z_slice, time_step = time_step, channel = channel, interpolation = True)
        except Exception as e:
            logging.info(f'An error occurred while processing {patient} full aorta: {e}')
            
    return 0

if __name__ == "__main__":
    # log sys_config 
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
        logging.info(f"Processing model {model}")
        for patient_type in class_labels:
            logging.info(f"Processing patient type {patient_type}")
            if patient_type.__contains__('compressed_sensing'):
                compressed_sensing_data = True
                suffix = '_compressed_sensing'
                patient_type = patient_type.replace('_compressed_sensing', '')
            else:
                compressed_sensing_data = False
                suffix = ''
            
            data_path = os.path.join(result_dir, model, patient_type+suffix)
            experiment_path = os.path.join('/',*data_path.split('/')[:-1])
            folder_save = os.path.join(experiment_path, 'CenterlinePostProcessingViz')
            
            create_centerlines(data_path, folder_save, patient_type = patient_type, compressed_sensing = compressed_sensing_data, cnn_predictions =True, z_slice = 15, time_step = 3, channel = 0)
            save_all_centerlines_in_one_image(folder_save, suffix = suffix)

    logging.info('Centerline extraction finished.')       


