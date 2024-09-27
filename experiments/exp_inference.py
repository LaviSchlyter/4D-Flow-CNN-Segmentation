# We use python as config, in order to allow for dynamic configurations and easy handle of lists
import os
import logging
import sys
import datetime
sys.path.append(os.path.join(os.getcwd(), 'lschlyter/CNN-segmentation'))
from utils import make_dir_safely
import model_zoo
# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Name of the experiments who's best model we want to use using a list
experiment_names_w_validation = ['231115-1723_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_size_40_finetune',
                    '231115-1727_da_0.25_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_size_40_finetune',
                    '231116-0910_da_0.0_nchan4_r2_loss_dice_e125_bs8_lr0.001_adBN_w_val_finetune',
                    '231116-1111_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_size_40_only_w_bern',
                    '231116-1114_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__w_val_tr_only_w_labels_only_w_bern'
                    ]
experiment_names_without_validation = ['231115-1731_da_0.25_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_finetune',
                                       '231116-0859_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_only_w_labels_finetune',
                                       '231116-0903_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_finetune',
                                       '231116-0925_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_only_w_labels_finetune',
                                       '231116-0937_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_only_w_labels_only_w_bern',
                                       '231116-1134_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_only_w_bern',
                                       '231116-1501_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_only_w_bern'
                                       ]


experiment_name = ['231116-1134_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__tr_size_40_only_w_bern']

experiment_names = experiment_name #experiment_names_without_validation, experiment_names_w_validation
# If using validation, you take the best model, if not, you take the one with longest epochs.
use_validation = False 
class_labels = ['controls', 'patients', 'patients_compressed_sensing', 'controls_compressed_sensing']
use_final_output_dir = True
nchannels = 4
predict_on_training = False
model_handle = model_zoo.UNet
slices = '40' # 'all' or '40'

