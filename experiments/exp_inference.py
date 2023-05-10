

# We give data: in h5py/numpy format
# We give the model name 
# Check that it exist else raise error
# The data is preprocessed in the same way as in the training (flag either 32 or all slices)
# Predict using the model
# Save the prediction in the same format as the input data
# Crop or pad to the orginal size of the input data
# When you predict, give the logits, softmax and argmax

import os
import logging
import sys
sys.path.append(os.path.join(os.getcwd(), 'lschlyter/CNN-segmentation'))
from utils import make_dir_safely
import model_zoo


# Name of the experiment who's best model we want to use
nchannels = 4
#experiment_name = f'unet3d_da_0.0nchannels{nchannels}_r1_loss_dice_cut_z_False_full_run_bern_only_w_labels'
experiment_name = 'unet3d_da_0.0nchannels4_r1_loss_crossentropy_cut_z_False_full_run_adaptive_batch_norm_lr_1e-4_e75_tr_27'
use_validation = False

model_handle = model_zoo.UNet
# Whether to use the all slices or 40 slices (check the training script cause it may also be the one only with labels)
slices = '40' # 'all' or '40'



