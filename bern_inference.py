import logging
import os
import h5py
import utils
import torch
import numpy as np
from losses import compute_dice
from utils import make_dir_safely
import re
# import experiment settings
from experiments import exp_inference as exp_config


def save_prediction(prediction, prediction_sized, output_path):
    logging.info('========================== Saving prediction to: {} ========================== '.format(output_path))
    # Save prediction
    #with h5py.File(output_path, 'w') as f:
    #    f.create_dataset('data', data=prediction.cpu())
    np.save(output_path.replace('.h5', '.npy'), prediction.cpu())
    logging.info('========================== Saved prediction to: {} ========================== '.format(output_path))

def run_inference(training_output_path, inference_input_path, label_path, inference_output_path, use_final_output_dir, final_model_output_file, patient_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('========================== Running inference on device: {} ========================== '.format(device))

    # ======================
    # log experiments settings
    # ======================
    logging.info('==============================================')
    logging.info('Running inference with the following settings:')
    
    logging.info('========================== Experiment name: {} ========================== '.format(exp_config.experiment_name))
    logging.info('========================== Prediction segmentation for image: {} ========================== '.format(inference_input_path))

    # ======================
    # load best model
    # ======================
    logging.info('========================== Looking for saved segmentation model... ========================== ')
    model_path = os.path.join(training_output_path, exp_config.experiment_name)
    # Not very robust 
    if exp_config.use_validation:
        best_model_path = os.path.join(model_path, list(filter(lambda x: 'best' in x, os.listdir(model_path)))[-1])
    else:
        best_model_path = os.path.join(model_path, os.listdir(model_path)[-1])
    if os.path.exists(best_model_path):
        logging.info('========================== Found best model from: {} ========================== '.format(best_model_path))
        # Load model
        model = exp_config.model_handle(in_channels=exp_config.nchannels, out_channels=2)
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Set model to eval mode
        model.eval()
    else:
        logging.warning('Did not find a model in: {}'.format(model_path))
        raise RuntimeError('No model avaialble for inference')
    
    # ======================
    # load data
    # ======================
    logging.info('========================== Loading data from: {} ========================== '.format(inference_input_path))
    # Load data
    image = h5py.File(inference_input_path,'r')["data"]
    logging.info('Shape of image before processing: {}'.format(image.shape))
    # Preprocess data
    image = utils.normalize_image(image)
    orig_volume_size = image.shape[0:4]
    # Crop or pad 


    if exp_config.slices == 'all':
        logging.info('========================== Using all slices ========================== ')
        common_image_shape = [144, 112, None, 48, 4] # [x, y, t, num_channels]
        image = utils.crop_or_pad_Bern_all_slices(image, common_image_shape)
    elif exp_config.slices == '40':
        logging.info('========================== Using 40 slices ========================== ')
        # Only use 40 slices
        common_image_shape = [144, 112, 40, 48, 4]
        image = utils.crop_or_pad_Bern_slices(image, common_image_shape)
    

    else:
        raise ValueError('Slices must be either all or 40')
    
    if exp_config.nchannels == 1:
        image = image[...,1:2]

        
    logging.info('Shape of image after processing: {}'.format(image.shape))
    
    image = torch.from_numpy(image).to(device=device, dtype=torch.float)
    
    # permute to [B, C, x, y, t]
    image = image.permute(2,4,0,1,3)
    logging.info('Shape of image after permuting: {}'.format(image.shape))

    # ======================
    # predict
    # ======================
    logging.info('========================== Predicting segmentation ========================== ')
    model.to(device)
    logits_b1, probs_b1, prediction_b1 = utils.predict(model, image[:image.shape[0]//4,...])
    logits_b2, probs_b2, prediction_b2 = utils.predict(model, image[image.shape[0]//4:(2*image.shape[0])//4,...])
    logits_b3, probs_b3, prediction_b3 = utils.predict(model, image[(2*image.shape[0])//4:(3*image.shape[0])//4,...])
    logits_b4, probs_b4, prediction_b4 = utils.predict(model, image[(3*image.shape[0])//4:,...])
    logits = torch.vstack((logits_b1,logits_b2, logits_b3, logits_b4))
    probs = torch.vstack((probs_b1,probs_b2, probs_b3, probs_b4))
    prediction = torch.vstack((prediction_b1,prediction_b2, prediction_b3, prediction_b4))
    #logits, probs, prediction = utils.predict(model, image)
    logging.info('Shape of prediction: {}'.format(prediction.shape))

    # Compute dice score if labels are available
    if os.path.exists(label_path):
        label = np.load(label_path)
        logging.info('Shape of label before processing: {}'.format(label.shape))
        if exp_config.slices == 'all':
            label = utils.crop_or_pad_Bern_all_slices(label, common_image_shape[:-1])
        elif exp_config.slices == '40':
            label = utils.crop_or_pad_Bern_slices(label, common_image_shape[:-1])

        label = torch.from_numpy(label).to(device=device, dtype=torch.long)
        label = torch.nn.functional.one_hot(label, num_classes = 2)
        label = label.transpose(1,4).transpose(0,2).transpose(3,4)
        logging.info('Shape of label after processing: {}'.format(label.shape))
        dice, mean_dice, mean_fg_dice = compute_dice(logits, label)
        logging.info('Dice score: {}'.format(dice))
        logging.info('Mean dice score: {}'.format(mean_dice))
        logging.info('Mean foreground dice score: {}'.format(mean_fg_dice))
        logging.info('Patient: {}'.format(patient_id))
        logging.info('Dice score: {}'.format(dice))
        logging.info('Mean dice score: {}'.format(mean_dice))
        logging.info('Mean foreground dice score: {}'.format(mean_fg_dice))
    

    # Crop or pad to original size
    prediction_sized = torch.unsqueeze(prediction, dim = -1)
    prediction_sized = prediction_sized.permute(4,1,2,0,3)
    prediction_sized = utils.crop_or_pad_4dvol(prediction_sized.cpu(), orig_volume_size)
    prediction_sized = prediction_sized.squeeze()
    logging.info('Shape of prediction_sized after cropping or padding: {}'.format(prediction_sized.shape))

    # Permute back to [x, y, z, t, C]
    prediction = prediction.permute(1,2,0,3)

    logging.info('Shape of prediction after permuting: {}'.format(prediction.shape))

    # ======================
    # save prediction
    # ======================
    if use_final_output_dir:
        logging.info('========================== Saving prediction to final folder: {} ========================== '.format(final_model_output_file))
        
        save_prediction(prediction, prediction_sized, final_model_output_file)
    else:
        logging.info('========================== Saving prediction to inference folder: {} ========================== '.format(inference_output_path))
        save_prediction(prediction, prediction_sized, inference_output_path)
    








            

if __name__ == '__main__':

    for class_label in ['controls']: # ['controls', 'patients']
    #class_label = 'controls' # controls or patients

        basepath = f'/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/{class_label}/h5'
        seg_basepath =f'/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/segmenter_rw_pw_hard/{class_label}'
        training_patients_ids = [re.split(r'seg_|.npy', x)[1] for x in os.listdir(seg_basepath)]
        patient_ids = os.listdir(basepath)


        for patient_id in patient_ids:
            patient_id = patient_id.split('.')[0]
            if not exp_config.predict_on_training:
                if training_patients_ids.__contains__(patient_id):
                    # The patient is on the training data
                    continue 
            training_output_path = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/logdir/"
            inference_input_path = f"{basepath}/{patient_id}.h5"
            label_path = f"{seg_basepath}/{class_label}/seg_{patient_id}.npy"
            inference_output_dir = f"/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/logdir/inference_results/{exp_config.experiment_name}/{class_label}"
            # This will be used when we want to save the model chosen for segmentation
            use_final_output_dir = False
            final_model_output_dir =f'/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/cnn_segmentations/{exp_config.experiment_name}/{class_label}'
            final_model_output_file = None
            if use_final_output_dir:
                utils.make_dir_safely(final_model_output_dir)
                final_model_output_file = os.path.join(final_model_output_dir, f"seg_{patient_id}.h5")
            make_dir_safely(inference_output_dir)
            inference_output_path = os.path.join(inference_output_dir, f"seg_{patient_id}.h5")


            run_inference(training_output_path, inference_input_path, label_path, inference_output_path, use_final_output_dir, final_model_output_file, patient_id)
