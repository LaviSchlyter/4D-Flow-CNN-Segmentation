# Same as notebook but in script form

import os 
import re
import h5py
import numpy as np



import SimpleITK as sitk
import math
from scipy import interpolate

from skimage.morphology import skeletonize_3d, dilation, cube, binary_erosion


from matplotlib import pyplot as plt

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from utils import normalize_image, normalize_image_new, make_dir_safely
from utils_centerline import (load_masked_data, load_cropped_data_sliced, extract_slice_from_sitk_image, load_masked_data_sliced)


def create_center_lines_final_segs(final_seg_path, img_path, hand_seg_path, suffix = '', skip = True, smoothness = 200, updated_ao = True, suffix_add=''):

        centerline_folder = 'SubjectCenterlines' + suffix + suffix_add
        interpolation_folder = 'SubjectCenterlines_and_interpolation' + suffix + suffix_add
        full_aorta_folder = 'SubjectCenterlines_and_interpolation_full_aorta' + suffix + suffix_add

        exp_path = os.path.join('/',*final_seg_path.split('/')[:-1])
        for folder in [centerline_folder, interpolation_folder, full_aorta_folder]:
            make_dir_safely(os.path.join(exp_path, folder))
        
        list_paths = os.listdir(final_seg_path)
        list_paths.sort()
        for n, patient in enumerate(list_paths):
            # Since some of the data does not have _.npy but rather .npy we need to take care of that
            name = patient.replace("seg_", "").replace("_.npy", "").replace(".npy", "")
            logging.info(f'Loading subject {n+1} out of {len(os.listdir(final_seg_path))}...')
            logging.info(f'Patient\'s name: {name}')

            # load the segmentation that was created with Nicolas's tool
            image = np.load(img_path + f'/{patient.replace("seg_", "")}')
            segmented = np.load(final_seg_path + f'/{patient}')


            # Check if the patient was hand-segmented or not
            if os.listdir(hand_seg_path).__contains__(patient):
                cnn_predictions = False
            else:
                cnn_predictions = True

            if cnn_predictions:
                points_ = skeleton_points(segmented, dilation_k = 0)
                points_dilated = skeleton_points(segmented, dilation_k = 4,erosion_k = 4)
            else:
                points_ = skeleton_points(segmented, dilation_k = 0)
                points_dilated = skeleton_points(segmented, dilation_k = 2,erosion_k = 2)
            points = points_dilated.copy()
            
            # ================================
            # VISUALIZATIONS
            # ================================

            fig = plt.figure('Centerline')
            fig, ax = plt.subplots(1, 2, figsize=(7, 7))
            axs = ax.ravel()
            axs[0].imshow(image[:,:,15,3,0], cmap='gray')
            axs[0].scatter(points_[:,1],points_[:,0], s=2, c='red', marker='o')
                
            axs[1].imshow(image[:,:,15,3,0], cmap='gray')
            axs[1].scatter(points_dilated[:, 1],points_dilated[:,0], s=2, c='red', marker='o')    
            
            name_save = exp_path + f'/{centerline_folder}/' + f'{name}' +'.png'
            logging.info(f'Saving image to {name_save}')    
            
            fig.savefig(name_save)
            plt.show()
            plt.close()

            # ================================
                
            if updated_ao:
                try:
                    points_order_ascending_aorta = order_points(points[::-1], angle_threshold=3/2*np.pi/2.)
                except Exception as e:
                    try:
                        points_order_ascending_aorta = order_points(points[::-1], angle_threshold=1/2*np.pi/2.)
                        print('points_order_ascending_aorta')
                    except Exception as e:
                        points_order_ascending_aorta = np.array([0,0,0])
                        logging.info(f'An error occurred while processing {patient} ascending aorta: {e}')
                    
                    
                points_limited = points[(points[:, 0] <= 90) & (points[:, 0] >= points_order_ascending_aorta[0][0]) & (points[:, 1] <= points_order_ascending_aorta[0][1])]
            else:
                ## Limit to sectors where ascending aorta is located
                points_limited = points[np.where(points[:,1]<65)]
                points_limited = points_limited[np.where(points_limited[:,0]<90)]
                points_limited = points_limited[points_limited[:,0].argsort()[::-1]]


            #points_limited = points_limited[2:-2]

            # Sort the points in ascending order with x
            #points_limited = points_limited[points_limited[:,0].argsort()[::-1]]

            if skip:
                temp_limited = []
                #modulo = 5 if len(points_limited) // 5 > 5 else 2 
                modulo = 2
                for index, element in enumerate(points_limited[2:]):
                    if (index%modulo)==0:
                        temp_limited.append(element)
                coords_limited = np.array(temp_limited)
            else:
                coords_limited = points_limited[2:].copy()
            #coords_limited are a bit confusing in order...
            x_limited = coords_limited[:,0]
            y_limited = coords_limited[:,1]
            z_limited = coords_limited[:,2]
            
            coords_limited = np.array([z_limited,y_limited,x_limited]).transpose([1,0])
            size_limited = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]

            # spline parametrization
            params_limited = [i / (size_limited[2] - 1) for i in range(size_limited[2])]
            tck_limited, _ = interpolate.splprep(np.swapaxes(coords_limited, 0, 1), k=3, s=smoothness)
            points_inter_limited = np.swapaxes(interpolate.splev(params_limited, tck_limited, der=0), 0, 1)
            fig1, ax = plt.subplots(1, 2, figsize=(7, 7))
            axs = ax.ravel()
            axs[0].imshow(image[:,:,15,3,0], cmap='gray')
            axs[0].scatter(points_limited[:,1],points_limited[:,0], s=2, c='red', marker='o')
            
            axs[1].imshow(image[:,:,15,3,0], cmap='gray')
            axs[1].scatter(points_inter_limited[:, 1],points_inter_limited[:,2], s=2, c='red', marker='o')    
            name_save = exp_path + f'/{interpolation_folder}/' + f'{name}' +'.png'
            fig1.savefig(name_save)
            plt.show()
            plt.close()


            # We try to order the points for the full aorta
            # If it doesn't work, we skip

            try:
                
                
                points_order = order_points(points)
                if skip:
                    temp = []
                    for index, element in enumerate(points_order[2:]):
                        if (index%2)==0:
                            temp.append(element)

                    coords = np.array(temp)
                    
                else:
                    coords = points_order[2:-2]
                

                x = coords[:,0]
                y = coords[:,1]
                z = coords[:,2]
                coords = np.array([z,y,x]).transpose([1,0])
                size = [36, 36, 256, 48, 4] # [x, y, z, t, num_channels]

                
                # spline parametrization for full aorta
                params = [i / (size[2] - 1) for i in range(size[2])]
                tck, _ = interpolate.splprep(np.swapaxes(coords, 0, 1), k=3, s=10)
                points_inter = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)

                fig, ax = plt.subplots(1, 2, figsize=(7, 7))
                axs = ax.ravel()
                axs[0].imshow(image[:,:,15,3,0], cmap='gray')
                axs[0].scatter(points_order[:,1],points_order[:,0], s=2, c='red', marker='o')
                
                axs[1].imshow(image[:,:,15,3,0], cmap='gray')
                axs[1].scatter(points_inter[:, 1],points_inter[:,2], s=2, c='red', marker='o')    
                name_save = exp_path + f'/{full_aorta_folder}/' + f'{name}' +'.png'
                plt.show()
                fig.savefig(name_save)
                plt.close()

            except Exception as e:
                logging.info(f'An error occurred while processing {patient} full aorta: {e}')
                
            
                
        
                





def interpolate_and_slice(image,
                          coords,
                          size, smoothness = 200):


    #coords are a bit confusing in order...
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    coords = np.array([z,y,x]).transpose([1,0])

    #convert the image to SITK (here let's use the intensity for now)
    sitk_image = sitk.GetImageFromArray(image[:,:,:])

    # spline parametrization
    params = [i / (size[2] - 1) for i in range(size[2])]
    tck, _ = interpolate.splprep(np.swapaxes(coords, 0, 1), k=3, s=smoothness)

    # derivative is tangent to the curve
    points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)
    Zs = np.swapaxes(interpolate.splev(params, tck, der=1), 0, 1)
    direc = np.array(sitk_image.GetDirection()[3:6])

    slices = []
    for i in range(len(Zs)):
        # I define the x'-vector as the projection of the y-vector onto the plane perpendicular to the spline
        xs = (direc - np.dot(direc, Zs[i]) / (np.power(np.linalg.norm(Zs[i]), 2)) * Zs[i])
        sitk_slice = extract_slice_from_sitk_image(sitk_image, points[i], Zs[i], xs, list(size[:2]) + [1], fill_value=0)
        np_image = sitk.GetArrayFromImage(sitk_slice).transpose(2, 1, 0)
        slices.append(np_image)

    # stick slices together
    return np.concatenate(slices, axis=2), points

# These functions are for interpolating the whole aorta
def nearest_neighbors(q,points,num_neighbors=2,exclude_self=True):
    d = ((points-q)**2).sum(axis=1)  # compute distances
    ndx = d.argsort() # indirect sort 
    start_ind = 1 if exclude_self else 0
    end_ind = start_ind+num_neighbors
    ret_inds = ndx[start_ind:end_ind]
    return ret_inds

def calc_angle(v1, v2, reflex=False):
    dot_prod = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    #round dot_prod for numerical stability
    angle = np.arccos(np.around(dot_prod,6))
    
    if (reflex == False):
        return angle
    else:
        return 2 * np.pi - angle
    
def order_points(candidate_points, angle_threshold = np.pi/2.):
    ordered_points = []
    
    #take first point
    ordered_points.append(candidate_points[0])
    nn = nearest_neighbors(ordered_points[-1], candidate_points,num_neighbors=1)
    #take second point
    ordered_points.append(candidate_points[nn[0]])
    #for ind,cp_i in enumerate(candidate_points):
    remove = 0
    while(len(ordered_points)<len(candidate_points)):
        
        #get 10 nearest neighbors of latest point
        nn = nearest_neighbors(ordered_points[-1], candidate_points,num_neighbors=10)
        # Taking the current point and the previous, we compute the angle to the current and eventual neighbourg
        # making sure its acute
        found = 0
        
        for cp_i in nn:
            ang = calc_angle(ordered_points[-2]-ordered_points[-1], candidate_points[cp_i]-ordered_points[-1])
            if ang > (angle_threshold):
                found =1

                ordered_points.append(candidate_points[cp_i])
            if found == 1:
                break 
        if found ==0:
            if remove >5:
                break
            
            candidate_points = list(candidate_points)
            candidate_points = [arr for arr in candidate_points if not np.array_equal(arr, ordered_points[-1])]
            candidate_points = np.array(candidate_points)
            ordered_points.pop()
            remove += 1
    ordered_points = np.array(ordered_points)

    return(ordered_points)

def skeleton_points(segmented, dilation_k=0, erosion_k = 0):
    # Average the segmentation over time (the geometry should be the same over time)
    avg = np.average(segmented, axis = 3)
    if dilation_k > 0:
        avg = binary_erosion(avg, selem=np.ones((erosion_k, erosion_k,erosion_k)))
        avg = dilation(avg, selem=np.ones((dilation_k, dilation_k,dilation_k)))
        
        
        
    # Compute the centerline points of the skeleton
    skeleton = skeletonize_3d(avg[:,:,:])
   
    # Get the points of the centerline as an array
    points = np.array(np.where(skeleton != 0)).transpose([1,0])

    # Order the points in ascending order with x
    points = points[points[:,0].argsort()[::-1]]
    
    return points
    

def create_center_lines(path, cnn_predictions = True, patient_type = 'controls', compressed_sensing_data = False, skip = True, smoothness = 200):

    suffix = '_compressed_sensing' if compressed_sensing_data else ''
    img_path = f'/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/{patient_type}/numpy{suffix}'

    
    logging.info(f'img path: {img_path}')


    centerline_folder = 'SubjectCenterlines' + suffix
    interpolation_folder = 'SubjectCenterlines_and_interpolation' + suffix
    full_aorta_folder = 'SubjectCenterlines_and_interpolation_full_aorta' + suffix
    
    # Create necessary directories
    if cnn_predictions:
        exp_path = os.path.join('/',*path.split('/')[:-1])
    else:
        exp_path = os.path.join('/',*path.split('/')[:-2],'hand_segmented_centerlines')
    for folder in [centerline_folder, interpolation_folder, full_aorta_folder]:
        make_dir_safely(os.path.join(exp_path, folder))

    i = 0
    for n, patient in enumerate(os.listdir(path)):
        # Since some of the data does not have _.npy but rather .npy we need to take care of that
        name = patient.replace("seg_", "").replace("_.npy", "").replace(".npy", "")
        
        logging.info(f'Loading subject {n+1} out of {len(os.listdir(path))}...')
        logging.info(f'Patient\'s name: {name}')

        # load the segmentation that was created with Nicolas's tool
        image = np.load(img_path + f'/{patient.replace("seg_", "")}')
        segmented = np.load(path + f'/{patient}')
        
        if cnn_predictions:
            points_ = skeleton_points(segmented, dilation_k = 0)
            points_dilated = skeleton_points(segmented, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmented, dilation_k = 0)
            points_dilated = skeleton_points(segmented, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()
        
        # ================================
        # VISUALIZATIONS
        # ================================

        fig = plt.figure('Centerline')
        fig, ax = plt.subplots(1, 2, figsize=(7, 7))
        axs = ax.ravel()
        axs[0].imshow(image[:,:,15,3,0], cmap='gray')
        axs[0].scatter(points_[:,1],points_[:,0], s=2, c='red', marker='o')
            
        axs[1].imshow(image[:,:,15,3,0], cmap='gray')
        axs[1].scatter(points_dilated[:, 1],points_dilated[:,0], s=2, c='red', marker='o')    
        
        name_save = exp_path + f'/{centerline_folder}/' + f'{name}' +'.png'
        logging.info(f'Saving image to {name_save}')    
        
        fig.savefig(name_save)
        plt.show()
        plt.close()

        # ================================
        # Limit to sectors where ascending aorta is located
        #points_limited = points[np.where(points[:,1]<65)]
        #points_limited = points_limited[np.where(points_limited[:,0]<90)]
#
        ## Order the points in ascending order with x
        #points_limited = points_limited[points_limited[:,0].argsort()[::-1]]        
        try:
            points_order_ascending_aorta = order_points(points[::-1], angle_threshold=3/2*np.pi/2.)
        except Exception as e:
            try:
                points_order_ascending_aorta = order_points(points[::-1], angle_threshold=1/2*np.pi/2.)
                
            except Exception as e:
                points_order_ascending_aorta = np.array([0,0,0])
                logging.info(f'An error occurred while processing {patient} ascending aorta: {e}')
        points_limited = points[(points[:, 0] <= 90) & (points[:, 0] >= points_order_ascending_aorta[0][0]) & (points[:, 1] <= points_order_ascending_aorta[0][1])]
        points_limited = points_limited[2:-2]

        # Sort the points in ascending order with x
        points_limited = points_limited[points_limited[:,0].argsort()[::-1]]

        if skip:
            temp_limited = []
            #modulo = 5 if len(points_limited) // 5 > 5 else 2 
            modulo = 2
            for index, element in enumerate(points_limited):
                if (index%modulo)==0:
                    temp_limited.append(element)
            coords_limited = np.array(temp_limited)
        else:
            coords_limited = points_limited.copy()
        #coords_limited are a bit confusing in order...
        x_limited = coords_limited[:,0]
        y_limited = coords_limited[:,1]
        z_limited = coords_limited[:,2]
        
        coords_limited = np.array([z_limited,y_limited,x_limited]).transpose([1,0])
        size_limited = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]

        # spline parametrization
        params_limited = [i / (size_limited[2] - 1) for i in range(size_limited[2])]
        tck_limited, _ = interpolate.splprep(np.swapaxes(coords_limited, 0, 1), k=3, s=smoothness)
        points_inter_limited = np.swapaxes(interpolate.splev(params_limited, tck_limited, der=0), 0, 1)
        fig1, ax = plt.subplots(1, 2, figsize=(7, 7))
        axs = ax.ravel()
        axs[0].imshow(image[:,:,15,3,0], cmap='gray')
        axs[0].scatter(points_limited[:,1],points_limited[:,0], s=2, c='red', marker='o')
        
        axs[1].imshow(image[:,:,15,3,0], cmap='gray')
        axs[1].scatter(points_inter_limited[:, 1],points_inter_limited[:,2], s=2, c='red', marker='o')    
        name_save = exp_path + f'/{interpolation_folder}/' + f'{name}' +'.png'
        fig1.savefig(name_save)
        plt.show()
        plt.close()


        # We try to order the points for the full aorta
        # If it doesn't work, we skip

        try:
            
            if skip:
                points_order = order_points(points)
                temp = []
                for index, element in enumerate(points_order[2:]):
                    if (index%2)==0:
                        temp.append(element)

                coords = np.array(temp)
            else:
                coords = points[2:-2]
            

            

            x = coords[:,0]
            y = coords[:,1]
            z = coords[:,2]
            coords = np.array([z,y,x]).transpose([1,0])
            size = [36, 36, 256, 48, 4] # [x, y, z, t, num_channels]

            
            # spline parametrization for full aorta
            params = [i / (size[2] - 1) for i in range(size[2])]
            tck, _ = interpolate.splprep(np.swapaxes(coords, 0, 1), k=3, s=10)
            points_inter = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)

            fig, ax = plt.subplots(1, 2, figsize=(7, 7))
            axs = ax.ravel()
            axs[0].imshow(image[:,:,15,3,0], cmap='gray')
            axs[0].scatter(points_order[:,1],points_order[:,0], s=2, c='red', marker='o')
            
            axs[1].imshow(image[:,:,15,3,0], cmap='gray')
            axs[1].scatter(points_inter[:, 1],points_inter[:,2], s=2, c='red', marker='o')    
            name_save = exp_path + f'/{full_aorta_folder}/' + f'{name}' +'.png'
            plt.show()
            fig.savefig(name_save)
            plt.close()

        except Exception as e:
            logging.info(f'An error occurred while processing {patient} full aorta: {e}')
            
        
            
            


    return 0


def save_all_in_one(model_path, hand_seg = False, suffix = ''):

    if hand_seg:
        save_images_path = os.path.join('/',*model_path.split('/')[:-2],'hand_segmented_centerlines') + f'/SubjectCenterlines{suffix}/'
    else:
        save_images_path = os.path.join('/',*model_path.split('/')[:-1])+ f'/SubjectCenterlines{suffix}/'
    final_seg_path


    fig, ax = plt.subplots(np.ceil(np.sqrt(n_images)).astype(int), np.ceil(np.sqrt(n_images)).astype(int), figsize=(20,20))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, file in enumerate(image_list):
        img = plt.imread(os.path.join(save_images_path, file))
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].imshow(img)
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].set_title(file)
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].axis('off')

    logging.info(f'Saving image to {os.path.join(save_images_path, f"all_images_centerlines{suffix}.png")}')
    
    plt.savefig(os.path.join(save_images_path, f'all_images_centerlines{suffix}.png'))
    plt.show()

def crop_or_pad_Bern_new(data, new_shape):

    processed_data = np.zeros(new_shape)
    # axis 0 is the x-axis and we crop from top since aorta is at the bottom
    # axis 1 is the y-axis and we crop equally from both sides
    # axis 2 is the z-axis and we crop from the right (end of the image) since aorta is at the left
    delta_axis0 = data.shape[0] - new_shape[0]
    delta_axis1 = data.shape[1] - new_shape[1]
    delta_axis2 = data.shape[2] - new_shape[2]
    if len(new_shape) == 5: # Image
        # The x is always cropped, y always padded, z_cropped
        processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:new_shape[1], :new_shape[2],...]

    if len(new_shape) == 4: # Label
        # The x is always cropped, y always padded, z_cropped
        processed_data[:, :data.shape[1],:,:data.shape[3],... ] = data[delta_axis0:,:new_shape[1], :new_shape[2],...]
    return processed_data

def prepare_and_write_masked_data_sliced_bern(model_path,
                            filepath_output,                    
                            patient_type,
                            load_anomalous=False, 
                            cnn_predictions = True,
                            suffix = ''):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]
    
    end_shape = [32, 32, 64, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    
    
    
    
    img_path = f'/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/{patient_type}/numpy{suffix}'
    seg_path = model_path
    
    patients = os.listdir(seg_path)
    num_images_to_load = len(patients)
    
    
    

    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [end_shape[2]*num_images_to_load,
                            end_shape[0],
                            end_shape[1],
                            end_shape[3],
                            end_shape[4]]

    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    dataset['sliced_images_%s' % patient_type] = hdf5_file.create_dataset("sliced_images_%s" % patient_type, images_dataset_shape, dtype='float32')
        
    i = 0
    for patient in patients: 
         
        logging.info(f'Loading subject {i+1} out of {num_images_to_load}...')
        logging.info(f'patient: ' + patient)

        
        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        segmented_original = np.load(os.path.join(seg_path, patient))
        

        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        segmented = dilation(segmented_original[:,:,:,3], cube(6))

        temp_for_stack = [segmented for i in range(time_steps)]
        segmented = np.stack(temp_for_stack, axis=3)

        # normalize image to -1 to 1
        image = normalize_image_new(image)
        seg_shape = list(segmented.shape)
        seg_shape.append(image.shape[-1])
        logging.info(f'Segmented shape: {seg_shape}')
        
        image = crop_or_pad_Bern_new(image, seg_shape)
        logging.info(f'Image shape after cropping and padding: {image.shape}')
        


        temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented

        # recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)


        

        if cnn_predictions:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()
        
        
        
        
        try:
            # Limit to sectors where ascending aorta is located
            #points = points[np.where(points[:,1]<65)]
            #points = points[np.where(points[:,0]<90)]
#
            ## Order the points in ascending order with x
            #points = points[points[:,0].argsort()[::-1]]
            
            points_order_ascending_aorta = order_points(points[::-1], angle_threshold=3/2*np.pi/2.)
            
            points = points[(points[:, 0] <= 90) & (points[:, 0] >= points_order_ascending_aorta[0][0]) & (points[:, 1] <= points_order_ascending_aorta[0][1])]
            points = points[2:-2]

            # Sort the points in ascending order with x
            points = points[points[:,0].argsort()[::-1]]

            temp = []
            for index, element in enumerate(points):
                if (index%2)==0:
                    temp.append(element)

            coords = np.array(temp)
            

            #===========================================================================================
            # Parameters for the interpolation and creation of the files

            # We create Slices across time and channels in a double for loop
            temp_for_channel_stacking = []
            for channel in range(image.shape[4]):

                temp_for_time_stacking = []
                for t in range(image.shape[3]):
                    straightened, interpolated_points = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape, smoothness = 10)
                    temp_for_time_stacking.append(straightened)

                channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
                temp_for_channel_stacking.append(channel_stacked)

            straightened = np.stack(temp_for_channel_stacking, axis=-1)
            image_out = straightened

            # make all images of the same shape
            logging.info(f'Image shape before cropping and padding: {image_out.shape}')
            
            image_out = crop_or_pad_Bern_new(image_out, end_shape)
            logging.info(f'Image shape after cropping and padding: {image_out.shape}')
            

            # move the z-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 2, 0)

            # add the image to the hdf5 file
            dataset['sliced_images_%s' % patient_type][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

            # increment the index being used to write in the hdf5 datasets
            i = i + 1
        except Exception as e:
            logging.info(f'An error occurred while processing {patient}: {e}')
            i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

def prepare_and_write_masked_data_sliced_bern_final_segs(img_path,
                        final_seg_path,
                        hand_seg_path,
                        filepath_output,                    
                        patient_type,
                        updated_ao = False,
                        skip = True,
                        smoothness = 10):


    common_image_shape = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]
    
    end_shape = [32, 32, 64, 48, 4]

    seg_path = final_seg_path
    patients = os.listdir(seg_path)
    num_images_to_load = len(patients)
    
    
    

    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [end_shape[2]*num_images_to_load,
                            end_shape[0],
                            end_shape[1],
                            end_shape[3],
                            end_shape[4]]

    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    dataset['sliced_images_%s' % patient_type] = hdf5_file.create_dataset("sliced_images_%s" % patient_type, images_dataset_shape, dtype='float32')

    i = 0
    for patient in patients:
        logging.info(f'Loading subject {i+1} out of {num_images_to_load}...')
        logging.info(f'patient: ' + patient)



        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        segmented_original = np.load(os.path.join(seg_path, patient))
        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        segmented = dilation(segmented_original[:,:,:,3], cube(6))
        

        temp_for_stack = [segmented for i in range(time_steps)]
        segmented = np.stack(temp_for_stack, axis=3)

        # normalize image to -1 to 1
        image = normalize_image_new(image)
        seg_shape = list(segmented.shape)
        seg_shape.append(image.shape[-1])
        logging.info(f'Segmented shape: {seg_shape}')
        
        image = crop_or_pad_Bern_new(image, seg_shape)
        logging.info(f'Image shape after cropping and padding: {image.shape}')
        


        temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented

        # recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        # check if patient was hand-segmented or not (check patients within hand_seg_path)
        if patient in os.listdir(hand_seg_path):
            cnn_predictions = False
        else:
            cnn_predictions = True

        if cnn_predictions:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 4,erosion_k = 4)
        else:
            points_ = skeleton_points(segmented_original, dilation_k = 0)
            points_dilated = skeleton_points(segmented_original, dilation_k = 2,erosion_k = 2)
        points = points_dilated.copy()

        
        
        if updated_ao:
            try:
            
                points_order_ascending_aorta = order_points(points[::-1], angle_threshold=3/2*np.pi/2.)
                logging.info('Points with angle threshold 3/2*np.pi/2.')
            except Exception as e:
                try:
                    points_order_ascending_aorta = order_points(points[::-1], angle_threshold=1/2*np.pi/2.)
                    logging.info('Points with angle threshold 1/2*np.pi/2.')
                except Exception as e:
                    points_order_ascending_aorta = np.array([0,0,0])
                    logging.info(f'An error occurred while processing {patient} ascending aorta: {e}')
            
            points = points[(points[:, 0] <= 90) & (points[:, 0] >= points_order_ascending_aorta[0][0]) & (points[:, 1] <= points_order_ascending_aorta[0][1])]

        else:
            points = points[np.where(points[:,1]<65)]
            points = points[np.where(points[:,0]<90)]
            points = points[points[:,0].argsort()[::-1]]
        
        #points = points[2:-2]
        #points = points[points[:,0].argsort()[::-1]]
        
        
        
        if skip:
            temp = []
            for index, element in enumerate(points[2:]):
                if (index%2)==0:
                    temp.append(element)

            coords = np.array(temp)
        else:
            coords = points[2:].copy()

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened, interpolated_points = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape, smoothness = smoothness)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)
        image_out = straightened

        # make all images of the same shape
        logging.info(f'Image shape before cropping and padding: {image_out.shape}')
        
        image_out = crop_or_pad_Bern_new(image_out, end_shape)
        logging.info(f'Image shape after cropping and padding: {image_out.shape}')
        

        # move the z-axis to the front, as we want to stack the data along this axis
        image_out = np.moveaxis(image_out, 2, 0)

        # add the image to the hdf5 file
        dataset['sliced_images_%s' % patient_type][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        # increment the index being used to write in the hdf5 datasets
        i = i + 1
        
    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0


def save_images_cropped_sliced(model_path,patient_type = 'controls', full_aorta = False, suffix = '', hand_seg = False):
    n_images = 64
    grid = 8
    size = 14
    aorta = ''
    if full_aorta:
        n_images = 256
        grid = 16
        size = 20
        aorta = '_full_aorta'
    if hand_seg:
        exp_path = os.path.join('/',*model_path.split('/')[:-2], 'hand_segmented_centerlines')
    else:
        exp_path = os.path.join('/',*model_path.split('/')[:-1])
    make_dir_safely(exp_path + f'/cropped_sliced{aorta+suffix}')
    print(exp_path)
    
    filepath_output = exp_path + f'/{patient_type+suffix}_sliced_images{aorta}.hdf5'
    sliced_data = h5py.File(filepath_output, 'r')
    images_cropped_sliced = sliced_data[f'sliced_images_{patient_type}']
    
    for n, p in enumerate(os.listdir(model_path)):
        patient = re.split(r'seg_|.npy', p)[1]
        fig, axs = plt.subplots(grid,grid, figsize=(size,size))
        ax = axs.ravel()
        for i in range(n_images):
            ax[i].imshow(images_cropped_sliced[i+(n*n_images), :,:, 3,1])
            #ax[i].set_title(i)
            ax[i].axis('off')
        plt.savefig(os.path.join(exp_path + f'/cropped_sliced{aorta+suffix}', f'{patient}_sliced{aorta}.png'))
        plt.close()
    sliced_data.close()
        
def save_images_cropped_sliced_masked(model_path,patient_type = 'controls', full_aorta = False, suffix = '', hand_seg = False):
    n_images = 64
    grid = 8
    size = 14
    aorta = ''
    if full_aorta:
        n_images = 256
        grid = 16
        size = 20
        aorta = '_full_aorta'
    
    if hand_seg:
        exp_path = os.path.join('/',*model_path.split('/')[:-2], 'hand_segmented_centerlines')
    else:
        exp_path = os.path.join('/',*model_path.split('/')[:-1])
    
    make_dir_safely(exp_path + f'/masked_cropped_sliced{aorta+suffix}')
    
    filepath_output = exp_path + f'/{patient_type+suffix}_masked_sliced_images{aorta}.hdf5'
    
    sliced_data = h5py.File(filepath_output, 'r')
    images_cropped_sliced = sliced_data[f'sliced_images_{patient_type}']
    print(images_cropped_sliced.shape)
    
    for n, p in enumerate(os.listdir(model_path)):
        print(p)
        patient = re.split(r'seg_|.npy', p)[1]
        fig, axs = plt.subplots(grid,grid, figsize=(size,size))
        ax = axs.ravel()
        for i in range(n_images):
            print
            ax[i].imshow(images_cropped_sliced[i+(n*n_images), :,:, 3,1])
            #ax[i].set_title(i)
            ax[i].axis('off')
        plt.savefig(os.path.join(exp_path + f'/masked_cropped_sliced{aorta+suffix}', f'{patient}_masked_sliced{aorta}.png'))
        plt.close()
    sliced_data.close()

def save_images_cropped_sliced_masked_final_segs(exp_path, final_seg_path, full_aorta = False, suffix = '', suffix_add = ''):

    n_images = 64
    grid = 8
    size = 14
    aorta = ''
    if full_aorta:
        n_images = 256
        grid = 16
        size = 20
        aorta = '_full_aorta'

    make_dir_safely(exp_path + f'/masked_cropped_sliced{aorta+suffix+suffix_add}')

    filepath_output = exp_path + f'/{patient_type+suffix+suffix_add}_masked_sliced_images{aorta}.hdf5'
    logging.info(f'Loading data from {filepath_output}')

    sliced_data = h5py.File(filepath_output, 'r')
    images_cropped_sliced = sliced_data[f'sliced_images_{patient_type}']
    print(images_cropped_sliced.shape)

    for n, p in enumerate(os.listdir(final_seg_path)):
        print(p)
        patient = re.split(r'seg_|.npy', p)[1]
        fig, axs = plt.subplots(grid,grid, figsize=(size,size))
        ax = axs.ravel()
        for i in range(n_images):
            print
            ax[i].imshow(images_cropped_sliced[i+(n*n_images), :,:, 3,1])
            #ax[i].set_title(i)
            ax[i].axis('off')
        plt.savefig(os.path.join(exp_path + f'/masked_cropped_sliced{aorta+suffix+suffix_add}', f'{patient}_masked_sliced{aorta}.png'))
        plt.close()



class_labels = ['controls']
#class_labels = ['controls', 'controls_compressed_sensing', 'patients', 'patients_compressed_sensing']


for patient_type in class_labels:

    if patient_type.__contains__('compressed_sensing'):
        compressed_sensing_data = True
        suffix = '_compressed_sensing'
        patient_type = patient_type.replace('_compressed_sensing', '')
    else:
        compressed_sensing_data = False
        suffix = ''
    logging.info(f'Running patient type {patient_type+suffix}')

    # Final segmentations
    final_seg_path = f'/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/final_segmentations/{patient_type+suffix}'

    all_seg_paths = [os.path.join(final_seg_path, f) for f in os.listdir(final_seg_path)]
    

    logging.info(f'Found {len(all_seg_paths)} segmentation files')


    # We need to rewrite the functions a bit because we need to know if it was hand-segmented or not

    img_path = f'/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/preprocessed/{patient_type}/numpy{suffix}'

    # Hand seg paths 
    hand_seg_path = f'/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady/segmenter_rw_pw_hard/{patient_type+suffix}'

    logging.info(f'img path: {img_path}')

    smoothness = 10
    skip_ = True
    updated_ao = True
    # The name needs to include smoothness, skip and weather it is updated or not
    extra_note = ''
    suffix_add = f'_smoothness_{smoothness}_skip_{skip_}_updated_ao_{updated_ao}' + extra_note

    # This is needed:
    #create_center_lines_final_segs(final_seg_path, img_path, hand_seg_path, suffix = suffix, skip= skip_, smoothness =smoothness, updated_ao= updated_ao, suffix_add=suffix_add)
    
    


    exp_path = os.path.join('/',*final_seg_path.split('/')[:-1])
    filepath_output = exp_path + f'/{patient_type+suffix+suffix_add}_masked_sliced_images.hdf5'


    """
    # Save the, in one image (to be uncommented if desired)
    save_images_path = os.path.join('/',*final_seg_path.split('/')[:-1])+ f'/SubjectCenterlines{suffix}/'
    make_dir_safely(save_images_path)
    image_list = os.listdir(save_images_path)
    image_list.sort()
    n_images = len(image_list)
    logging.info(f'Found {n_images} images')
    fig, ax = plt.subplots(np.ceil(np.sqrt(n_images)).astype(int), np.ceil(np.sqrt(n_images)).astype(int), figsize=(20,20))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, file in enumerate(image_list):
        img = plt.imread(os.path.join(save_images_path, file))
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].imshow(img)
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].set_title(file)
        ax[i//np.ceil(np.sqrt(n_images)).astype(int), i%np.ceil(np.sqrt(n_images)).astype(int)].axis('off')

    logging.info(f'Saving image to {os.path.join(save_images_path, f"all_images_centerlines{suffix}.png")}')
    
    plt.savefig(os.path.join(save_images_path, f'all_images_centerlines{suffix}.png'))
    plt.show()
    

    """

    
    if not os.path.exists(filepath_output):
        masked_sliced_data = prepare_and_write_masked_data_sliced_bern_final_segs(img_path=img_path, final_seg_path=final_seg_path, hand_seg_path=hand_seg_path, filepath_output=filepath_output, patient_type=patient_type, updated_ao= updated_ao, skip= skip_, smoothness =smoothness)
        masked_sliced_data = h5py.File(filepath_output, 'r')
    else:
        logging.info(f'File {filepath_output} already exists')
        masked_sliced_data = h5py.File(filepath_output, 'r')
#
    masked_images_cropped_sliced = masked_sliced_data[f'sliced_images_{patient_type}']

    save_images_cropped_sliced_masked_final_segs(exp_path, final_seg_path, suffix= suffix, suffix_add=suffix_add)

    
    
    """
    
    #save_all_in_one(model_path=model_path, hand_seg= False, suffix = suffix)
    #save_all_in_one(model_path=hand_seg_path, hand_seg= True, suffix = suffix)
    exp_path = os.path.join('/',*all_seg_paths[0].split('/')[:-1])
    #hand_path = os.path.join('/',*hand_seg_path.split('/')[:-2],'hand_segmented_centerlines')
    filepath_output = exp_path + f'/{patient_type+suffix}_masked_sliced_images.hdf5'

    if not os.path.exists(filepath_output):
        masked_sliced_data = prepare_and_write_masked_data_sliced_bern_final_segs(img_path=img_path, final_seg_path=final_seg_path, hand_seg_path=hand_seg_path, filepath_output=filepath_output, patient_type=patient_type)
        masked_sliced_data = h5py.File(filepath_output, 'r')
    else:
        logging.info(f'File {filepath_output} already exists')
        masked_sliced_data = h5py.File(filepath_output, 'r')

        # This is for the hand-segmented patients
    #filepath_output_hand = hand_path + f'/{patient_type+suffix}_masked_sliced_images.hdf5'
    #if not os.path.exists(filepath_output_hand):
    #    hand_masked_sliced_data = prepare_and_write_masked_data_sliced_bern(model_path=hand_seg_path, filepath_output=filepath_output_hand, patient_type=patient_type, cnn_predictions= False, suffix= suffix)
    #    hand_masked_sliced_data = h5py.File(filepath_output_hand, 'r')
    #else:
    #    hand_masked_sliced_data = h5py.File(filepath_output_hand, 'r')

    #masked_images_cropped_sliced = masked_sliced_data[f'sliced_images_{patient_type}']
    #hand_masked_images_cropped_sliced = hand_masked_sliced_data[f'sliced_images_{patient_type}']

    #save_images_cropped_sliced_masked(model_path, patient_type = patient_type, suffix= suffix)
    #save_images_cropped_sliced_masked(hand_seg_path, patient_type = patient_type, suffix= suffix, hand_seg= True)
    
    """