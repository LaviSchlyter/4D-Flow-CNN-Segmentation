# We take the manually segmented aorta's and do the pre-processing before putting them into a h5 file

import os
import h5py
import numpy as np
from utils import crop_or_pad_Bern_slices, crop_or_pad_Bern_all_slices, normalize_image

def process_patient_data(patient, image_data, label_data, common_image_shape, common_label_shape, mode='fixed_slices'):
    """
    Processes the data for a single patient, handling both imaging and segmentation data.

    Parameters:
    - patient (type): Description about patient parameter.
    - image_data (numpy.ndarray): The patient's imaging data.
    - label_data (numpy.ndarray): The patient's segmentation or label data.
    - common_image_shape (tuple): The standardized shape for processed image data.
    - common_label_shape (tuple): The standardized shape for processed label data.
    - mode (str): Mode of operation, either 'fixed_slices' or 'all_slices'.

    Returns:
    tuple: Processed imaging and segmentation data for the patient.
    """
    #image_data = np.load(os.path.join(img_path, patient.replace("seg_", "")))
    image_data = normalize_image(image_data)

    #label_data = np.load(os.path.join(seg_path, patient))

    if mode == 'fixed_slices':
        image_data = crop_or_pad_Bern_slices(image_data, common_image_shape)
        label_data = crop_or_pad_Bern_slices(label_data, common_label_shape)
        
    elif mode == 'all_slices':
        image_data = crop_or_pad_Bern_all_slices(image_data, common_image_shape)
        label_data = crop_or_pad_Bern_all_slices(label_data, common_label_shape)


    image_data = np.moveaxis(image_data, 2, 0)
    label_data = np.moveaxis(label_data, 2, 0)
    label_data = label_data.astype(np.uint8)

    return image_data, label_data#, index_w_labels

def prepare_and_write_data_bern(idx_start, idx_end, filepath_output, train_test, mode='fixed_slices', z_slices=40, with_labels=True):
    """
    Populates an HDF5 file with processed imaging and segmentation data of provided patients.

    Parameters:
    - idx_start (int): The starting index of the patients to process.
    - idx_end (int): The ending index of the patients to process.
    - filepath_output (str): The filepath to the HDF5 file to populate.
    - train_test (str): The train/test string to know which dataset to populate.
    - mode (str): Mode of operation, either 'fixed_slices' or 'all_slices'.
    - z_slices (int): The number of slices to use for the z axis.
    - with_labels (bool): Whether to include labels or not.

    """
    # Common logic setup
    common_image_shape = [144, 112, z_slices if mode == 'fixed_slices' else None, 48, 4]
    common_label_shape = [144, 112, z_slices if mode == 'fixed_slices' else None, 48]
    basepath_jerem = os.getcwd()
    hand_seg_path_controls = basepath_jerem + '/data/inselspital/kady/segmenter_rw_pw_hard/controls'
    hand_seg_path_patients = basepath_jerem + '/data/inselspital/kady/segmenter_rw_pw_hard/patients'
    list_hand_seg_images = os.listdir(hand_seg_path_controls) + os.listdir(hand_seg_path_patients) + os.listdir(hand_seg_path_patients + '_compressed_sensing') + os.listdir(hand_seg_path_controls+'_compressed_sensing')
    list_hand_seg_images.sort()

    # randomize list_hand_seg_images alwayzs the same way
    # TODO ? Maybe later as we might have to introduce some patients in the training and not just validation set
    rng = np.random.RandomState(0)
    rng.shuffle(list_hand_seg_images)
    


    #seg_path = os.path.join(basepath_jerem, 'data/inselspital/kady/segmenter_rw_pw_hard', 'controls')
    #img_path = os.path.join(basepath_jerem, 'data/inselspital/kady/preprocessed', 'controls', 'numpy')
    patients = list_hand_seg_images[idx_start:idx_end]
    # Sort the list
    num_images_to_load = len(patients)
    
    print("Output filepath: ", filepath_output)
    hdf5_file = h5py.File(filepath_output, "a" if mode == 'all_slices' else "w")
    
    if mode == 'fixed_slices':
        images_dataset_shape = [common_image_shape[2] * num_images_to_load,
                                common_image_shape[0],
                                common_image_shape[1],
                                common_image_shape[3],
                                common_image_shape[4]]
        labels_dataset_shape = [common_label_shape[2] * num_images_to_load,
                                common_label_shape[0],
                                common_label_shape[1],
                                common_label_shape[3]]
        
        
        dataset = {
            f'images_{train_test}': hdf5_file.create_dataset(f"images_{train_test}", images_dataset_shape, dtype='float32'),
            f'labels_{train_test}': hdf5_file.create_dataset(f"labels_{train_test}", labels_dataset_shape, dtype='uint8')
        }
    else:  # all_slices
        dataset = {
            f'images_{train_test}': hdf5_file.create_dataset(f'images_{train_test}', shape=(0, common_image_shape[0], common_image_shape[1], common_image_shape[3], common_image_shape[4]), maxshape=(None, common_image_shape[0], common_image_shape[1], common_image_shape[3], common_image_shape[4]), dtype='float32'),
            f'labels_{train_test}': hdf5_file.create_dataset(f'labels_{train_test}', shape=(0, common_image_shape[0], common_image_shape[1], common_image_shape[3]), maxshape=(None, common_image_shape[0], common_image_shape[1], common_image_shape[3]), dtype='uint8'),
            'alias': hdf5_file.create_dataset('alias', shape=(0,), maxshape=(None,), dtype='uint8')
        }

    for i, patient in enumerate(patients):
        print(f'loading subject {i} out of {num_images_to_load}...')
        print(f'patient: {patient}')
        CS = False

        if os.listdir(hand_seg_path_controls).__contains__(patient):
            seg_path = os.path.join(basepath_jerem, 'data/inselspital/kady/segmenter_rw_pw_hard', 'controls')
            img_path = os.path.join(basepath_jerem, 'data/inselspital/kady/preprocessed', 'controls', 'numpy')
        elif os.listdir(hand_seg_path_patients).__contains__(patient):
            seg_path = os.path.join(basepath_jerem, 'data/inselspital/kady/segmenter_rw_pw_hard', 'patients')
            img_path = os.path.join(basepath_jerem, 'data/inselspital/kady/preprocessed', 'patients', 'numpy')
        elif os.listdir(hand_seg_path_controls+'_compressed_sensing').__contains__(patient):
            seg_path = os.path.join(basepath_jerem, 'data/inselspital/kady/segmenter_rw_pw_hard', 'controls_compressed_sensing')
            img_path = os.path.join(basepath_jerem, 'data/inselspital/kady/preprocessed', 'controls', 'numpy_compressed_sensing')
            CS = True
        elif os.listdir(hand_seg_path_patients + '_compressed_sensing').__contains__(patient):
            seg_path = os.path.join(basepath_jerem, 'data/inselspital/kady/segmenter_rw_pw_hard', 'patients_compressed_sensing')
            img_path = os.path.join(basepath_jerem, 'data/inselspital/kady/preprocessed', 'patients', 'numpy_compressed_sensing')
            CS = True
        
        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        seg = np.load(os.path.join(seg_path, patient))

        if CS:
            # Remove the first 10 slices
            image = image[10:]
            seg = seg[10:]

        
        image_data, label_data = process_patient_data(patient, image, seg, common_image_shape, common_label_shape, mode)
        
        if mode == 'fixed_slices':
            dataset[f'images_{train_test}'][i*common_image_shape[2]:(i+1)*common_image_shape[2], :, :, :, :] = image_data
            dataset[f'labels_{train_test}'][i*common_label_shape[2]:(i+1)*common_label_shape[2], :, :, :] = label_data
        else:  # all_slices
            #
            if with_labels:
                index_w_labels = np.where(label_data.sum(axis = (1,2,3)) > 0)[0]
                label_data = label_data[index_w_labels]
            print(label_data.shape)
            print(image_data.shape)
            dataset['labels_%s' % train_test].resize(dataset['labels_%s' % train_test].shape[0] +label_data.shape[0], axis=0)
            dataset['labels_%s' % train_test][-label_data.shape[0]:] = label_data

            # add the image to the hdf5 file
            dataset['images_%s' % train_test].resize(dataset['images_%s' % train_test].shape[0] +label_data.shape[0], axis=0)    
            if with_labels:
                dataset['images_%s' % train_test][-label_data.shape[0]:] = image_data[index_w_labels]
            else:
                dataset['images_%s' % train_test][-label_data.shape[0]:] = image_data

            # add the alias to the hdf5 file
            dataset['alias'].resize(dataset['alias'].shape[0] +label_data.shape[0], axis=0)
            alias = np.ones(image_data.shape[0])
            alias[3:-3] = 0
            if with_labels:
                dataset['alias'][-label_data.shape[0]:] = alias[index_w_labels]
            else:
                dataset['alias'][-label_data.shape[0]:] = alias

    hdf5_file.close()
def prepare_data(basepath, config):
    """
    Prepare data based on a configuration.

    Parameters:
    - basepath (str): The base path where the data is located.
    - config (dict): Configuration for preparing and writing data.
    """
    filepath_output = os.path.join(basepath, 'data', config['filename'])
    
    if config['flag']:
        prepare_and_write_data_bern(
            config['idx_start'], 
            config['idx_end'], 
            filepath_output, 
            config['train_test'], 
            mode=config['mode'],
            with_labels=config.get('with_labels', True), # Defaulting to True for with_labels
            z_slices=config.get('z_slices', None)
        )

# Init main
if __name__ == '__main__':

    basepath = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation"

    configurations = [{
        'flag': False, # prepare_train_fixed_slices
        'filename': f'size_40_bern_images_and_labels_from_01_to_35.hdf5',
        'idx_start': 0,
        'idx_end': 35,
        'train_test': 'train',
        'mode': 'fixed_slices',
        'z_slices': 40
    },
    {
        'flag': False, # prepare_validation_fixed_slices
        'filename': f'size_40_bern_images_and_labels_from_36_to_45.hdf5',
        'idx_start': 35,
        'idx_end': 45,
        'train_test': 'validation',
        'mode': 'fixed_slices',
        'z_slices': 40
    },
    {
        'flag': False, 
        'filename': f'size_40_bern_images_and_labels_from_01_to_45.hdf5',
        'idx_start': 0,
        'idx_end': 45,
        'train_test': 'train',
        'mode': 'fixed_slices',
        'z_slices': 40
    },
    {
        'flag': False, # prepare_train_all_slices_w_labels
        'filename': f'only_w_labels_bern_images_and_labels_from_01_to_35.hdf5',
        'idx_start': 0,
        'idx_end': 35,
        'train_test': 'train',
        'mode': 'all_slices',
        'with_labels': True
    },
    {
        'flag': False, # prepare_validation_all_slices_w_labels
        'filename': f'only_w_labels_bern_images_and_labels_from_36_to_45.hdf5',
        'idx_start': 35,
        'idx_end': 45,
        'train_test': 'validation',
        'mode': 'all_slices',
        'with_labels': True
    },
    {
        'flag': True, # prepare_validation_all_slices_w_labels
        'filename': f'only_w_labels_bern_images_and_labels_from_0_to_45.hdf5',
        'idx_start': 0,
        'idx_end': 45,
        'train_test': 'train',
        'mode': 'all_slices',
        'with_labels': True
    },
    {
        'flag': False, # prepare_train_all_slices
        'filename': f'bern_images_and_labels_from_01_to_35.hdf5',
        'idx_start': 0,
        'idx_end': 35,
        'train_test': 'train',
        'mode': 'all_slices',
        'with_labels': False
    },
    {
        'flag': False, # prepare_validation_all_slices
        'filename': f'bern_images_and_labels_from_36_to_45.hdf5',
        'idx_start': 35,
        'idx_end': 45,
        'train_test': 'validation',
        'mode': 'all_slices',
        'with_labels': False
    },
    {
        'flag': False, # prepare_train_fixed_slices
        'filename': f'size_40_bern_images_and_labels_from_101_to_127.hdf5',
        'idx_start': 0,
        'idx_end': 27,
        'train_test': 'train',
        'mode': 'fixed_slices',
        'z_slices': 40
    },
    {
        'flag': False, # prepare_validation_fixed_slices
        'filename': f'size_40_bern_images_and_labels_from_122_to_127.hdf5',
        'idx_start': 21,
        'idx_end': 27,
        'train_test': 'validation',
        'mode': 'fixed_slices',
        'z_slices': 40
    },
    {
        'flag': False, # prepare_validation_fixed_slices
        'filename': f'size_40_bern_images_and_labels_from_101_to_127.hdf5',
        'idx_start': 0,
        'idx_end': 27,
        'train_test': 'train',
        'mode': 'fixed_slices',
        'z_slices': 40
    },
    {
        'flag': False, # prepare_train_all_slices_w_labels
        'filename': f'only_w_labels_bern_images_and_labels_from_101_to_121.hdf5',
        'idx_start': 0,
        'idx_end': 21,
        'train_test': 'train',
        'mode': 'all_slices',
        'with_labels': True
    },
    {
        'flag': False, # prepare_validation_all_slices_w_labels
        'filename': f'only_w_labels_bern_images_and_labels_from_122_to_127.hdf5',
        'idx_start': 21,
        'idx_end': 27,
        'train_test': 'validation',
        'mode': 'all_slices',
        'with_labels': True
    },
    {
        'flag': False, # prepare_validation_all_slices_w_labels
        'filename': f'only_w_labels_bern_images_and_labels_from_101_to_127.hdf5',
        'idx_start': 0,
        'idx_end': 27,
        'train_test': 'train',
        'mode': 'all_slices',
        'with_labels': True
    },
    {
        'flag': False, # prepare_train_all_slices
        'filename': f'bern_images_and_labels_from_101_to_121.hdf5',
        'idx_start': 0,
        'idx_end': 21,
        'train_test': 'train',
        'mode': 'all_slices',
        'with_labels': False
    },
    {
        'flag': False, # prepare_validation_all_slices
        'filename': f'bern_images_and_labels_from_122_to_127.hdf5',
        'idx_start': 21,
        'idx_end': 27,
        'train_test': 'validation',
        'mode': 'all_slices',
        'with_labels': False
    }]
    
    
    # Iterate over configurations to prepare data
    for config in configurations:
        prepare_data(basepath, config)





"""

    # Prepare train data for fixed slices
    prepare_train_fixed_slices = False
    idx_start = 0
    idx_end = 21
    z_slices = 40
    train_test = 'train'
    filepath_output = os.path.join(basepath, 'data', f'size_{z_slices}_bern_images_and_labels_from_101_to_121.hdf5')
    if prepare_train_fixed_slices:
        prepare_and_write_data_bern(basepath, idx_start, idx_end, filepath_output, train_test, mode='fixed_slices', z_slices=z_slices)

    # Prepare validation data for fixed slices
    prepare_validation_fixed_slices = False
    idx_start = 22
    idx_end = 27
    z_slices = 40
    train_test = 'validation'
    filepath_output = os.path.join(basepath, 'data', f'size_{z_slices}_bern_images_and_labels_from_101_to_121.hdf5')
    if prepare_validation_fixed_slices:
        prepare_and_write_data_bern(basepath, idx_start, idx_end, filepath_output, train_test, mode='fixed_slices', z_slices=z_slices)

    # Prepare train data for all slices
    # Only with labels
    prepare_train_all_slices_w_labels = False
    with_labels = False
    idx_start = 0
    idx_end = 21
    train_test = 'train'
    filepath_output = os.path.join(basepath, 'data', f'only_w_labels_bern_images_and_labels_from_101_to_121.hdf5')
    if prepare_train_all_slices_w_labels:
        prepare_and_write_data_bern(basepath, idx_start, idx_end, filepath_output, train_test, mode='all_slices', with_labels=with_labels)

    # Prepare validation data for all slices
    # Only with labels
    prepare_validation_all_slices_w_labels = False
    with_labels = True
    idx_start = 22
    idx_end = 27
    train_test = 'validation'
    filepath_output = os.path.join(basepath, 'data', f'only_w_labels_bern_images_and_labels_from_122_to_127.hdf5')
    if prepare_validation_all_slices_w_labels:
        prepare_and_write_data_bern(basepath, idx_start, idx_end, filepath_output, train_test, mode='all_slices', with_labels=with_labels)

    # Prepare train data for all slices
    prepare_train_all_slices = False
    with_labels = False
    idx_start = 0
    idx_end = 21
    train_test = 'train'
    filepath_output = os.path.join(basepath, 'data', f'bern_images_and_labels_from_101_to_121.hdf5')
    if prepare_train_all_slices:
        prepare_and_write_data_bern(basepath, idx_start, idx_end, filepath_output, train_test, mode='all_slices', with_labels=with_labels)
    
    # Prepare validation data for all slices
    prepare_validation_all_slices = False
    with_labels = False
    idx_start = 22
    idx_end = 27
    train_test = 'validation'
    filepath_output = os.path.join(basepath, 'data', f'bern_images_and_labels_from_122_to_127.hdf5')
    if prepare_validation_all_slices:
        prepare_and_write_data_bern(basepath, idx_start, idx_end, filepath_output, train_test, mode='all_slices', with_labels=with_labels)
"""





    