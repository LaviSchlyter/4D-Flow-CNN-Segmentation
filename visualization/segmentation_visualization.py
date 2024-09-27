import os 
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation')

from mayavi import mlab as mlab

from utils import make_dir_safely

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

basepath_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/logdir/inference_results'

experiment_list =['231115-1731_da_0.25_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_finetune',
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
#expriment_list = ['231115-1723_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_size_40_finetune']

experiment_list.sort()
def visualize_and_save(patient_name, seg, save_images_path, compressed_sensing=False):
        mlab.figure()
        mlab.contour3d(seg[...,0], colormap='gray')
        mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')  # Display axis
        mlab.orientation_axes()
        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=focalpoint)
        if compressed_sensing:
            save_images_path = os.path.join(save_images_path, 'compressed_sensing')
            make_dir_safely(save_images_path)
        mlab.savefig(os.path.join(save_images_path, f'{patient_name}_seg_3d.png'))
        mlab.close()

def plot_and_save_images(image_folder, save_path, image_title):
    logging.info(image_folder, os.listdir(image_folder))
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
    plt.show()

for expriment in experiment_list:
    logging.info(f'Processing experiment {expriment}')
    results_path = os.path.join(basepath_path, expriment)
    save_images_path = os.path.join(results_path, 'images')
    make_dir_safely(save_images_path)
    class_results_files = []
    class_results_paths = []

    class_labels = ['controls', 'patients','patients_compressed_sensing', 'controls_compressed_sensing'] #['controls', 'patients','patients_compressed_sensing', 'controls_compressed_sensing']
    
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

        n_images = len(image_list)


        # Process main images folder
        plot_and_save_images(save_images_path, results_path, 'all_inference_segmentations_viz.png')

        # Process compressed_sensing subfolder
        compressed_sensing_folder = os.path.join(save_images_path, 'compressed_sensing')
        plot_and_save_images(compressed_sensing_folder, results_path, 'all_compressed_sensing_segmentations_viz.png')

        


