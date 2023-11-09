

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
import datetime
sys.path.append(os.path.join(os.getcwd(), 'lschlyter/CNN-segmentation'))
from utils import make_dir_safely
import model_zoo
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# Name of the experiment who's best model we want to use
experiment_name = '231030-1542_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001_w_val_tr_only_w_labels_bern_and_freiburg'
#231030-1608_da_0.0_nchan4_r1_loss_dice_e80_bs8_lr0.001_w_val_tr_only_w_labels_finetune
#231030-1542_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001_w_val_tr_only_w_labels_bern_and_freiburg

# Decide which data to use 
class_labels = ['patients_compressed_sensing' ]#['controls', 'patients', 'patients_compressed_sensing', 'controls_compressed_sensing']

# This will be used when we want to save the model chosen for segmentation
use_final_output_dir = False
nchannels = 4
use_validation = True
predict_on_training = False

model_handle = model_zoo.UNet
# Whether to use the all slices or 40 slices (check the training script cause it may also be the one only with labels)
slices = '40' # 'all' or '40'



