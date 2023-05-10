import model_zoo
import torch

# ======================================================================
# brief description of this experiment
# ======================================================================
# ...

# ======================================================================
# Model settings
# ======================================================================
model_handle = model_zoo.UNet
use_adaptive_batch_norm = False # If True set train_with_Bern to False

train_with_bern = True
defrozen_conv_blocks = False

# If cut_z is True, we cut the first and last 3 slices of the z axis
cut_z = False
cut_z_saved = 3
run_number = 1
da_ratio = 0.0
nchannels = 1 # [intensity, vx, vy, vz]
#note = f'_bern_{train_with_bern}_adaptive_{use_adaptive_batch_norm}_DEBUG'
note = f'_full_run'
extra_info = "_finetune_lr_1e-3_e10_AdamW_40_slices"

use_saved_model = True
with_validation = True
train_file_name = 'size_40_bern_images_and_labels_from_101_to_121' # bern_images_and_labels_from_101_to_116, only_w_labels_bern_images_and_labels_from_101_to_116, size_32_bern_images_and_labels_from_101_to_104
# If with_validation is True, we use the following file for validation
# else it's loaded but not used
val_file_name = 'size_40_bern_images_and_labels_from_122_to_127'
add_pixels_weight = 100
# ======================================================================
# data settings
# ======================================================================
data_mode = '3D'
image_size = [144, 112, 48] # [x, y, time]
nlabels = 2 # [background, foreground]


# ======================================================================
# training settings
# ======================================================================
#max_steps = 10000
#steps = 1000
epochs = 10
batch_size = 8
learning_rate = 1e-3
optimizer_handle = torch.optim.AdamW
betas = (0.9, 0.98)
loss_type = 'dice'  # crossentropy/dice
summary_writing_frequency = 20 # 4 times in each epoch (if n_tr_images=20, batch_size=8)
train_eval_frequency = 500 # every 4 epochs (if n_tr_images=20, batch_size=8)
val_eval_frequency = 500 # every 4 epochs (if n_tr_images=20, batch_size=8)
save_frequency = 200 # every 10 epochs (if n_tr_images=20, batch_size=8)

continue_run = False
debug = True
augment_data = False
experiment_name_saved_model = 'unet3d_da_' + str(da_ratio) + 'nchannels' + str(nchannels) + '_r' + str(run_number) + '_loss_' + loss_type + '_cut_z_' + str(cut_z_saved) + note 
experiment_name = 'unet3d_da_' + str(da_ratio) + 'nchannels' + str(nchannels) + '_r' + str(run_number) + '_loss_' + loss_type + '_cut_z_' + str(cut_z) + note + extra_info


# ======================================================================
# test settings NOT MINE
# ======================================================================
# iteration number to be loaded after training the model (setting this to zero will load the model with the best validation dice score)
#load_this_iter = 0
#batch_size_test = 4
#save_qualitative_results = True
#save_results_subscript = 'initial_training_unet'
#save_inference_result_subscript = 'inference_out_unet'