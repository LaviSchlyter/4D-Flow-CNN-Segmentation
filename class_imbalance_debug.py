
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
import sys
import model_zoo
import data_freiburg_numpy_to_hdf5
from utils import make_dir_safely, normalize_image
from losses import compute_dice


def test(images, labels, batch_size):
    """
    Function to create mini batches from the dataset of a certain batch size
    :param images: input images
    :param labels: labels
    :param batch_size: batch size
    :return: mini batches"""
    assert len(images) == len(labels)
    
    # Generate randomly selected slices in each minibatch

    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # Use only fraction of the batches in each epoch

    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue


        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]
        
        # ===========================
        # check if the velocity fields are to be used for the segmentation...
        # ===========================

        
        yield X, y





if __name__ == '__main__':
    # Bern data
    # This has already done Bern_numpy_to_hdf5.py
    basepath = "/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady"
    bern_tr = h5py.File(basepath + '/bern_images_and_labels_from_101_to_104.hdf5','r')
    bern_vl = h5py.File(basepath + '/bern_images_and_labels_from_105_to_106.hdf5','r')
    images_tr = bern_tr['images_train']
    labels_tr = bern_tr['labels_train']
    images_vl = bern_vl['images_validation']
    labels_vl = bern_vl['labels_validation']        
    # Bern data
    
    basepath = "/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady"
    res_bern_tr = h5py.File(basepath + '/size_32_bern_images_and_labels_from_101_to_104.hdf5','r')
    res_bern_vl = h5py.File(basepath + '/size_32_bern_images_and_labels_from_105_to_106.hdf5','r')
    res_images_tr = res_bern_tr['images_train']
    res_labels_tr = res_bern_tr['labels_train']
    res_images_vl = res_bern_vl['images_validation']
    res_labels_vl = res_bern_vl['labels_validation']      
    res_gen = test(res_images_tr, res_labels_tr, 8)  
    print(res_gen.__next__())
    
