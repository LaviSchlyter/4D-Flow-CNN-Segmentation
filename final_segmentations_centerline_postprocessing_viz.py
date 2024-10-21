import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
import numpy as np
import matplotlib.pyplot as plt

# Define base paths for your projects
basepath = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/"
# Append the path for CNN-segmentation
cnn_segmentation_path = os.path.join(basepath, 'CNN-segmentation/')
sys.path.insert(0, cnn_segmentation_path)
from config import system as sys_config
from src.inference.cnn_seg_centerline_extraction import create_centerlines, save_all_centerlines_in_one_image
from src.inference.cnn_seg_cross_sectional_slices import save_images_cropped_sliced_masked, prepare_and_write_masked_data_sliced_bern
sys.path.remove(cnn_segmentation_path)

if __name__ == '__main__':

    class_labels = ['controls', 'patients', 'controls_compressed_sensing', 'patients_compressed_sensing']
    
    for patient_type in class_labels:
        if patient_type.__contains__('compressed_sensing'):
            compressed_sensing_data = True
            suffix = '_compressed_sensing'
            patient_type = patient_type.replace('_compressed_sensing', '')
        else:
            compressed_sensing_data = False
            suffix = ''
        final_seg_path = f'{sys_config.project_data_root}/segmentations/final_segmentations/{patient_type+suffix}'
        save_path = os.path.join('/',*final_seg_path.split('/')[:-1])
        folder_save = os.path.join(save_path, 'FinalCenterlinePostProcessing')

        logging.info(f'Running subject class {patient_type+suffix} containing {len(os.listdir(final_seg_path))} subjects')
        # Here we create the centerlines for the final-segmented data and save them
        # TODO: currently makes no difference cnn_predictions = False or True, because I realise for the postprocessiing of the final data - I used the same dilation and erosion kernel
        create_centerlines(final_seg_path, folder_save, patient_type = patient_type, compressed_sensing = compressed_sensing_data, cnn_predictions = False)
        save_all_centerlines_in_one_image(folder_save, suffix = suffix)

        # Here we preprocess the slices and visualize them.
        filepath_output = os.path.join(folder_save, f'{patient_type+suffix}_masked_sliced_images.hdf5')
        if not os.path.exists(filepath_output):
            logging.info(f'File {filepath_output} does not exist. Preparing and writing masked data.')
            masked_sliced_data = prepare_and_write_masked_data_sliced_bern(final_seg_path, filepath_output, patient_type = patient_type, cnn_predictions = False, suffix = suffix)
        else:
            logging.info(f'File {filepath_output} already exists')

        save_images_cropped_sliced_masked(final_seg_path, folder_save, folder_save, patient_type = patient_type, full_aorta = False, suffix = suffix)

    logging.info('Finished running the final segmented post-processing of the centerlines.')



