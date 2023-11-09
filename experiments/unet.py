import model_zoo
import torch
import datetime

# ======================================================================
# Model settings
# ======================================================================
model_handle = model_zoo.UNet

"""
We have several ways of training the segmentation network:
1. Training a model only with the Freiburg data (use_saved_model = False and train_with_bern = False)
2. Training with Bern data from scratch without Freiburg (use_saved_model = False, train_with_bern = True, only_w_bern = True)
3. Finetuning with batch normalization for Bern data with saved model (use_saved_model = True, use_adaptive_batch_norm = True)
4. Traininig with Bern and Freiburg data (use_saved_model = False, train_with_bern = True, only_w_bern = False)
5. Finetune with Bern but with all layers (use_saved_model = True, train_with_bern = True)
"""
REPRODUCE = False
# If you want to reproduce the results, set REPRODUCE to True and use the following settings:
SEED = 0

use_adaptive_batch_norm = False  # If True, train_with_Bern should be False
train_with_bern = False
use_saved_model = False
only_w_bern = False # You can choose if you want to only train with Bern or mix both
with_validation = True
defrozen_conv_blocks = False

# Data settings
data_mode = '3D'
image_size = [144, 112, 48]  # [x, y, time]
nlabels = 2  # [background, foreground]
train_file_name = 'only_w_labels_bern_images_and_labels_from_01_to_45' # size_40_bern_images_and_labels_from_01_to_35, only_w_labels_bern_images_and_labels_from_101_to_121, size_40_bern_images_and_labels_from_101_to_121
val_file_name = 'only_w_labels_bern_images_and_labels_from_36_to_45'
add_pixels_weight = 100


# Training settings
cut_z = True  # If True, cut the first and last 3 slices of the z axis
cut_z_saved = 3
run_number = 1
da_ratio = 0.25
nchannels = 4  # [intensity, vx, vy, vz]
epochs = 125
batch_size = 8
learning_rate = 1e-3

optimizer_handle = torch.optim.AdamW
betas = (0.9, 0.98)
loss_type = 'dice' # 'dice' or 'crossentropy'

summary_writing_frequency = 50
train_eval_frequency = 200
val_eval_frequency = 200
save_frequency = 200
continue_run = False
augment_data = False

# Timestamp and Experiment naming
timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
da_tag = f'da_{da_ratio}'
cutz_tag = '_cutz' if cut_z else ''
channel_tag = f'nchan{nchannels}'
run_tag = f'r{run_number}'
loss_tag = f'loss_{loss_type}'
epoch_tag = f'e{epochs}'
batch_tag = f'bs{batch_size}'
lr_tag = f'lr{learning_rate}'
full_run_freiburg = '_full_run_freiburg' if not use_saved_model and not train_with_bern else ''
adaptive_bn_tag = 'adBN' if use_adaptive_batch_norm else ''
w_validation_tag = '_w_val' if with_validation else ''
# If training with Bern, extract note from train_file_name
if train_with_bern:
    if 'only_w_labels' in train_file_name:
        note = '_tr_only_w_labels'
    elif 'size_40' in train_file_name:
        note = '_tr_size_40'
else:
    note = ''
if only_w_bern:
    note += '_only_w_bern'

if use_saved_model:
    note += '_finetune'
if not use_saved_model and train_with_bern and not only_w_bern:
    note += '_bern_and_freiburg'
extra_info = ''
experiment_name = f'{timestamp}_{da_tag}_{channel_tag}_{run_tag}_{loss_tag}{cutz_tag}_{epoch_tag}_{batch_tag}_{lr_tag}_{adaptive_bn_tag}{full_run_freiburg}{w_validation_tag}{note}{extra_info}'

# unfortuneately, we need to hardcode the name of the saved model
#experiment_name_saved_model = '231023-1021_da_0.0_nchan4_r1_loss_dice_e125_bs8_lr0.001__full_run_freiburg'  # Placeholder for saved model naming if needed
#experiment_name_saved_model = '231023-1024_da_0.0_nchan4_r1_loss_dice_cutz_e125_bs8_lr0.001__full_run_freiburg'
#experiment_name_saved_model = '231023-1025_da_0.0_nchan4_r1_loss_cross_entropy_e125_bs8_lr0.001__full_run_freiburg'
#experiment_name_saved_model = '231023-1030_da_0.0_nchan4_r1_loss_crossentropy_e125_bs8_lr0.001__full_run_freiburg'
#experiment_name_saved_model = '231023-1031_da_0.0_nchan4_r1_loss_crossentropy_cutz_e125_bs8_lr0.001__full_run_freiburg'
experiment_name_saved_model = '231023-1024_da_0.0_nchan4_r1_loss_dice_cutz_e125_bs8_lr0.001__full_run_freiburg'


