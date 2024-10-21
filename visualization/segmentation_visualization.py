import os
import numpy as np
from pyvirtualdisplay import Display
from mayavi import mlab
mlab.options.offscreen = True  
import matplotlib.pyplot as plt
import sys
import argparse
import yaml
import logging

# Set up virtual display before any rendering libraries are imported or used
display = Display(visible=0, size=(800, 600))
display.start()

# Set up environment variables to ensure Qt and VTK use the virtual display
# Explicitly disable the 'xcb' plugin and force offscreen rendering
os.environ["DISPLAY"] = display.new_display_var
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Ensure no X server is used
os.environ["MPLBACKEND"] = "Agg"             # Use Agg backend for matplotlib


# Import additional modules after setting the environment variables
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation')
from src.helpers.utils import make_dir_safely, gather_experiment_paths

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

basepath_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/Results'

def visualize_and_save(patient_name, seg, save_images_path, compressed_sensing=False):
    fig = mlab.figure()  # Create a figure
    fig.scene.off_screen_rendering = True  # Set offscreen rendering for this figure
    mlab.contour3d(seg[..., 0], colormap='gray')
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')  # Display axis
    mlab.orientation_axes()
    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=focalpoint)
    if compressed_sensing:
        save_images_path = os.path.join(save_images_path, 'compressed_sensing')
        make_dir_safely(save_images_path)
    mlab.savefig(os.path.join(save_images_path, f'{patient_name}_seg_3d.png'))
    mlab.close()

def plot_and_save_images(image_folder, save_path, image_title):
    # If image folder is empty, return
    if not os.listdir(image_folder):
        logging.info(f'Folder {image_folder} is empty')
        return
    image_list = [img for img in os.listdir(image_folder) if img.endswith('.png')]
    n_images = len(image_list)

    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    ax = ax.flatten()

    for i, file in enumerate(image_list):
        img = plt.imread(os.path.join(image_folder, file))
        ax[i].imshow(img)
        ax[i].set_title(file)
        ax[i].axis('off')

    for j in range(i + 1, len(ax)):
        ax[j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(save_path, image_title))
    

# The following long list was during experimental phase
default_experiment_list =['231115-1731_da_0.25_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_finetune',
                    '231116-0859_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_only_w_labels_finetune',
                    '231116-0903_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_finetune',
                    '231116-0925_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_only_w_labels_finetune',
                    '231116-0937_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_only_w_labels_only_w_bern',
                    '231116-1134_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_only_w_bern',
                    '231116-1501_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_only_w_bern',
                    '231115-1723_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_size_40_finetune',
                    '231115-1727_da_0.25_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_size_40_finetune',
                    '231116-0910_da_0.0_nchan4_r2_loss_dice_e125_bs8_lr0.001_adBN_w_val_finetune',
                    '231116-1111_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_size_40_only_w_bern',
                    '231116-1114_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_only_w_labels_only_w_bern',
                    
                    ]
default_experiment_list = ['241017-0948_da_0.0_nchan4_loss_dice_e5_bs8_lr1e-3__tr_size_40_only_w_bern_opt_AdamW_seed_0',
    '231116-1134_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_only_w_bern']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with specific settings.")
    parser.add_argument('--config_path', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Check if configuration file is provided else use the default experiment list
    if args.config_path:
        with open(args.config_path, 'r') as file:
            config = yaml.safe_load(file)
            experiment_names_path = gather_experiment_paths(config['models_dir'], config['filters']['specific_times'])
            # Extract experiment list from config
            experiment_list = [os.path.basename(exp) for exp in experiment_names_path]
    else:
        # Use the default experiment list defined in the script
        experiment_list = default_experiment_list

    experiment_list.sort()



    for experiment in experiment_list:
        logging.info(f'Processing experiment {experiment}')
        results_path = os.path.join(basepath_path, experiment)
        save_images_path = os.path.join(results_path, 'prediction_images')
        make_dir_safely(save_images_path)
        class_results_files = []
        class_results_paths = []

        class_labels = ['controls', 'patients','patients_compressed_sensing', 'controls_compressed_sensing'] 
        
        for class_label in class_labels:
            try:
                for file in os.listdir(os.path.join(results_path, class_label)):
                    if file.endswith('.npy'):
                        class_results_files.append(file)
                        class_results_paths.append(os.path.join(results_path, class_label, file))
            except:
                pass

            azimuth = 270
            elevation = 220
            distance = 300
            focalpoint = (80, 70, 10)

        
        logging.info(f'class_results_files {class_results_files}')
        for n, patient_n in enumerate(class_results_files):
            logging.info(f'Processing patient {patient_n}')
            patient_name = patient_n.split('_.npy')[0]
            patient_name = patient_name.split('.npy')[0]

            
            seg = np.load(class_results_paths[n], allow_pickle=True)
            # Check if compressed sensing
            if class_results_paths[n].__contains__('compressed_sensing'):
                visualize_and_save(patient_name, seg, save_images_path, compressed_sensing=True)
            else:
                visualize_and_save(patient_name, seg, save_images_path)

        
        image_list = os.listdir(save_images_path)
        image_list.sort()
        # Take only png files
        image_list = [img for img in image_list if img.endswith('.png')]
        logging.info(f'image_list {image_list}')
        n_images = len(image_list)


        # Process main images folder
        plot_and_save_images(save_images_path, results_path, 'all_inference_segmentations_viz.png')
        
        # Process compressed_sensing subfolder
        compressed_sensing_folder = os.path.join(save_images_path, 'compressed_sensing')
        make_dir_safely(compressed_sensing_folder)
        plot_and_save_images(compressed_sensing_folder, results_path, 'all_compressed_sensing_segmentations_viz.png')

    logging.info('\n\n --- All visualizations saved successfully.---')
    display.stop()


    


