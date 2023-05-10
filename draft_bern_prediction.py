import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import model_zoo
import gc
import data_freiburg_numpy_to_hdf5
from utils import make_dir_safely, normalize_image

print("Starting script...")

def crop_or_pad_Bern(data, new_shape):
    processed_data = np.zeros(new_shape)
    # axis 0 is the x-axis and we crop from top since aorta is at the bottom
    # axis 1 is the y-axis and we crop equally from both sides
    # axis 2 is the z-axis and we crop from the right (end of the image) since aorta is at the left
    delta_axis0 = data.shape[0] - new_shape[0]
    delta_axis1 = data.shape[1] - new_shape[1]
    delta_axis2 = data.shape[2] - new_shape[2]
    if len(new_shape) == 5: # Image
        # The x is always cropped, y always padded, z_cropped
        processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],...]

    if len(new_shape) == 4: # Label
        # The x is always cropped, y always padded, z_cropped
        processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:, :new_shape[2],...]
    return processed_data
    

# Bern segmetation
path_bern_seg = "/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/segmenter_rw_pw_hard/controls"
path_bern_img = "/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/controls/numpy"
print(os.listdir(path_bern_seg))

bern_images = []
bern_segmentations = []
for patient in os.listdir(path_bern_seg)[:1]:
    print(patient)

    img_seg = np.load(os.path.join(path_bern_seg, patient))
    img = np.load(os.path.join(path_bern_img, patient.replace("seg_", "")))
    img = normalize_image(img)

    img_processed = crop_or_pad_Bern(img, (144,112,32,48, 4))
    seg_processed = crop_or_pad_Bern(img_seg, (144,112,32,48))
    # shapes: (144, 112, 32, 48, c)
    img_processed = img_processed.transpose(2,0,1,3,4)
    seg_processed = seg_processed.transpose(2,0,1,3)
    # shapes: (32, 144, 112, 48, c) [per patient]
    bern_images.append(img_processed)
    bern_segmentations.append(seg_processed)
    
# load saved model
loss = "dice"
out_channels = 2
in_channels = 1
run = 1
note = '_full_run'
cut_z = 3
da = 0.0
model_path = f'/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/logdir/unet3d_da_{da}nchannels{in_channels}_r{run}_loss_{loss}_cut_z_{cut_z}{note}'
#model_path = f'/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/logdir/unet3d_da_{da}nchannels{in_channels}_r_phase_dice_cut_z_5_debug'
#model_path = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/logdir/unet3d_da_0.0nchannels1_r_DEBUG_LOSS_phase_dice_cut_z_0"
best_model_path = os.path.join(model_path, list(filter(lambda x: 'best' in x, os.listdir(model_path)))[-1])
print(best_model_path)
model = model_zoo.UNet(in_channels, out_channels)
model.load_state_dict(torch.load(best_model_path))
model.eval()

make_dir_safely(model_path + "/results/Bern/inputs")
make_dir_safely(model_path + "/results/Bern/labels")
make_dir_safely(model_path + "/results/Bern/predictions")

save_bern = model_path + "/results/Bern/"
for input_bern, label_bern, patient_id in zip(bern_images, bern_segmentations, os.listdir(path_bern_seg)[:1]):
    with torch.no_grad():
        print(patient_id)
        input_ = torch.from_numpy(input_bern)
        # Input (batch_size, channell,x,y,t)
        input_.transpose_(1,4).transpose_(2,4).transpose_(3,4)
        labels = torch.from_numpy(label_bern)

        print("start cut")
        input_ = input_[cut_z:-cut_z, ...]
        print("End cut")
        if in_channels == 1:
            # We take the phase x channel
            input_ = input_[:,1:2,...]
        print("Enter model")
        pred = model(input_.float())
        print("Exit model")
        
        np.save(os.path.join(save_bern + "/inputs", patient_id), input_.detach().numpy())  
        np.save(os.path.join(save_bern + "/labels", patient_id), labels.detach().numpy())
        np.save(os.path.join(save_bern + "/predictions", patient_id), pred.detach().numpy())
        print("Finished patient")
    


