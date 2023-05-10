import h5py
import numpy as np
from utils import normalize_image, crop_or_pad_Bern_slices, crop_or_pad_4dvol
import os
from skimage.morphology import skeletonize_3d, dilation, cube
import SimpleITK as sitk
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import math 
"""
def crop_or_pad_zeros(data, new_shape):

    processed_data = np.zeros(new_shape)

    # ======================
    # axes 0 (x) and 1 (y) need to be cropped or not
    # ======================
    # axis 0 will be cropped only from the top (as the aorta often exists until the bottom end of the image)
    # axis 1 will be cropped evenly from the right and left
    # ======================
    # axes 2 (z) and 3 (t) will be padded with zeros if the original image is smaller
    # ======================

    delta_axis0 = data.shape[0] - new_shape[0]
    delta_axis1 = data.shape[1] - new_shape[1]

    if len(new_shape) is 5: # image
        if delta_axis1 > 0:
            processed_data[:, :, :data.shape[2], :data.shape[3], :] = data[delta_axis0:, (delta_axis1//2):-(delta_axis1//2), :, :, :]
        elif delta_axis1 == 0:
            processed_data[:, :, :data.shape[2], :data.shape[3], :] = data[delta_axis0:, :, :, :, :]
        else:
            delta_axis1 = abs(delta_axis1)
            processed_data[:, (delta_axis1//2):-(delta_axis1//2), :data.shape[2], :data.shape[3], :] = data[delta_axis0:, :, :, :, :]

    elif len(new_shape) is 4: # label
        if delta_axis1 > 0:
            processed_data[:, :, :data.shape[2], :data.shape[3]] = data[delta_axis0:, (delta_axis1//2):-(delta_axis1//2), :]
        elif delta_axis1 == 0:
            processed_data[:, :, :data.shape[2], :data.shape[3]] = data[delta_axis0:, :, :, :]
        else:
            delta_axis1 = abs(delta_axis1)
            processed_data[:, (delta_axis1//2):-(delta_axis1//2), :data.shape[2], :data.shape[3]] = data[delta_axis0:, :, :, :]

        #processed_data[:, :, :data.shape[2], :data.shape[3]] = data[delta_axis0:, (delta_axis1//2):-(delta_axis1//2), :, :]

    return processed_data
def crop_or_pad_Bern_slices(data, new_shape):
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
"""
def crop_or_pad_Bern_new(data, new_shape):
    print('Check this function!')
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


def pad_z_dim(data, new_shape):
    processed_data = np.zeros(new_shape)
    if len(new_shape) == 5: # Image
    
        processed_data[:, :, :data.shape[2], :, :] = data
    if len(new_shape) == 4: # Label
        processed_data[:, :, :data.shape[2], :] = data
    return processed_data

def crop_or_pad_4dvol_along_2(vol, n):    
    x = vol.shape[2]
    x_s = (x - n) // 2
    x_c = (n - x) // 2
    if len(vol.shape) == 5:
        if x > n: # original volume has more slices that the required number of slices
            vol_cropped = vol[:, :, x_s:x_s + n, :, :]
        else: # original volume has equal of fewer slices that the required number of slices
            vol_cropped = np.zeros((vol.shape[0], vol.shape[1], n, vol.shape[3], vol.shape[4]))
            vol_cropped[:, :, x_c:x_c + x, :, :] = vol
    elif len(vol.shape) == 4:
        if x > n:
            vol_cropped = vol[:, :, x_s:x_s + n, :]
        else:
            vol_cropped = np.zeros((vol.shape[0], vol.shape[1], n, vol.shape[3]))
            vol_cropped[:, :, x_c:x_c + x, :] = vol
    return vol_cropped


def normalize_image_new(image):

    # ===============
    # initialize with zeros
    # ===============
    normalized_image = np.zeros((image.shape))

    # ===============
    # normalize magnitude channel
    # ===============
    normalized_image[...,0] = image[...,0] / np.amax(image[...,0])

    # ===============
    # normalize velocities
    # ===============

    # extract the velocities in the 3 directions
    velocity_image = np.array(image[...,1:4])

    # denoise the velocity vectors
    velocity_image_denoised = gaussian_filter(velocity_image, 0.5)

    # compute per-pixel velocity magnitude
    velocity_mag_image = np.linalg.norm(velocity_image_denoised, axis=-1)

    # velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
    # find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    # vpercentile_min = np.percentile(velocity_mag_image, 5)
    # vpercentile_max = np.percentile(velocity_mag_image, 95)

    normalized_image[...,1] = 2.*(velocity_image_denoised[...,0] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
    normalized_image[...,2] = 2.*(velocity_image_denoised[...,1] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
    normalized_image[...,3] = 2.*(velocity_image_denoised[...,2] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1


    return normalized_image




# ====================================================================================
# MASKED DATA
#====================================================================================
def prepare_and_write_masked_data(basepath,
                           filepath_output,
                           train_test):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    # For Bern the max sizes are:
    # x: 144, y: 112, z: 64, t: 33
    #common_image_shape = [144, 112, 64, 33, 4] # [x, y, z, t, num_channels]
    #common_label_shape = [144, 112, 64, 33] # [x, y, z, t]
    common_image_shape = [144, 112, 40, 48, 4] # [x, y, z, t, num_channels]
    common_label_shape = [144, 112, 40, 48] # [x, y, z, t]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    
    seg_path = basepath + '/segmenter_rw_pw_hard/controls'
    img_path = basepath + '/preprocessed/controls/numpy'
    #num_images_to_load = idx_end + 1 - idx_start
    if train_test == 'train':
        patients = os.listdir(seg_path)[:4]
        num_images_to_load = len(patients)
    elif train_test == 'validation':
        patients = os.listdir(seg_path)[4:]
        num_images_to_load = len(patients)
    
    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [common_image_shape[2]*num_images_to_load,
                            common_image_shape[0],
                            common_image_shape[1],
                            common_image_shape[3],
                            common_image_shape[4]]

    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    dataset['masked_images_%s' % train_test] = hdf5_file.create_dataset("masked_images_%s" % train_test, images_dataset_shape, dtype='float32')
    #dataset['labels_%s' % train_test] = hdf5_file.create_dataset("labels_%s" % train_test, labels_dataset_shape, dtype='uint8')

    i = 0
    for patient in patients: 
        
        #print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        print('loading subject ' + str(i) + ' out of ' + str(num_images_to_load) + '...')
        
        
        # load the numpy image (saved by the dicom2numpy file)
        #if load_anomalous:
        #    image_data = np.load(basepath + '/' + 'anomalous_subject' + '/image.npy')
        #else:
        #    image_data = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy')

        image_data = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        
        # normalize the image
        image_data = normalize_image_new(image_data)
        # make all images of the same shape
        image_data = crop_or_pad_Bern_slices(image_data, common_image_shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        image_data = np.moveaxis(image_data, 2, 0)


        # load the numpy label (saved by the random walker segmenter)
        #if load_anomalous:
        #    label_data = np.load(basepath + '/' + 'anomalous_subject' + '/random_walker_prediction.npy')
        #else:
        #    label_data = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/random_walker_prediction.npy')
        label_data = np.load(os.path.join(seg_path, patient))
        # make all images of the same shape
        label_data = crop_or_pad_Bern_slices(label_data, common_label_shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        label_data = np.moveaxis(label_data, 2, 0)
        # cast labels as uints
        label_data = label_data.astype(np.uint8)


        temp_images_intensity = image_data[:,:,:,:,0] * label_data
        temp_images_vx = image_data[:,:,:,:,1] * label_data
        temp_images_vy = image_data[:,:,:,:,2] * label_data
        temp_images_vz = image_data[:,:,:,:,3] * label_data

        #recombine the images
        image_data = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        # add the image to the hdf5 file
        dataset['masked_images_%s' % train_test][i*common_image_shape[2]:(i+1)*common_image_shape[2], :, :, :, :] = image_data

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_masked_data(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    dataset_filepath = basepath + '/masked_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_masked_data(basepath = basepath,
                               filepath_output = dataset_filepath,
                               
                               train_test = train_test)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

def prepare_and_write_masked_data_patient(basepath,
                           filepath_output,
                           train_test):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    # For Bern the max sizes are:
    # x: 144, y: 112, z: 64, t: 33 (but because of network we keep 48)
    #common_image_shape = [144, 112, 64, 48, 4] # [x, y, z, t, num_channels]
    #common_label_shape = [144, 112, 64, 48] # [x, y, z, t]
    # We first change the input image to the network size image so that it has the same as the segmentation and then we modify both
    network_common_image_shape = [144, 112, None, 48, 4] # [x, y, t, num_channels]
    common_image_shape = [144, 112, 40, 48, 4] # [x, y, z, t, num_channels]
    common_label_shape = [144, 112, 40, 48] # [x, y, z, t]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    image_path_base = "/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady"
    img_path = image_path_base + '/preprocessed/controls/numpy'

    patients = os.listdir(basepath)
    #patients = patients[:2]
    # The basepath is the model directory
    num_images_to_load = len(patients)


    
    # ==========================================
    # we will stack all images along their z-axis
    # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
    # ==========================================
    images_dataset_shape = [common_image_shape[2]*num_images_to_load,
                            common_image_shape[0],
                            common_image_shape[1],
                            common_image_shape[3],
                            common_image_shape[4]]

    # ==========================================
    # create a hdf5 file
    # ==========================================
    dataset = {}
    hdf5_file = h5py.File(filepath_output, "w")

    # ==========================================
    # write each subject's image and label data in the hdf5 file
    # ==========================================
    dataset['masked_images_%s' % train_test] = hdf5_file.create_dataset("masked_images_%s" % train_test, images_dataset_shape, dtype='float32')
    #dataset['labels_%s' % train_test] = hdf5_file.create_dataset("labels_%s" % train_test, labels_dataset_shape, dtype='uint8')

    i = 0
    for patient in patients: 
        
        #print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        print('loading subject ' + str(i) + ' out of ' + str(num_images_to_load) + '...')
        print(patient)
        
        image_data = np.load(os.path.join(img_path, patient.replace("_seg_cnn", "")))
        
        # normalize the image
        image_data = normalize_image_new(image_data)
        print('Shape of image before network resizing',image_data.shape)
        # The images need to be sized as in the input of network
        
        
        # make all images of the same shape
        image_data = crop_or_pad_Bern_slices(image_data, common_image_shape)
        print('Shape of image after network resizing',image_data.shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        image_data = np.moveaxis(image_data, 2, 0)

        label_data = np.load(os.path.join(basepath, patient))
        # make all images of the same shape
        label_data = crop_or_pad_Bern_slices(label_data, common_label_shape)
        # move the z-axis to the front, as we want to concantenate data along this axis
        label_data = np.moveaxis(label_data, 2, 0)
        # cast labels as uints
        label_data = label_data.astype(np.uint8)


        temp_images_intensity = image_data[:,:,:,:,0] * label_data
        temp_images_vx = image_data[:,:,:,:,1] * label_data
        temp_images_vy = image_data[:,:,:,:,2] * label_data
        temp_images_vz = image_data[:,:,:,:,3] * label_data

        #recombine the images
        image_data = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)

        # add the image to the hdf5 file
        dataset['masked_images_%s' % train_test][i*common_image_shape[2]:(i+1)*common_image_shape[2], :, :, :, :] = image_data

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_masked_data_patient(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    dataset_filepath = os.path.abspath(os.path.join(basepath, os.pardir)) + '/masked_images_from_' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_masked_data_patient(basepath = basepath,
                               filepath_output = dataset_filepath,
                               
                               train_test = train_test)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

# ====================================================================================
# *** MASKED DATA END ****
#=====================================================================================


# ====================================================================================

def show_center_lines(basepath):
    
    seg_path = basepath + '/segmenter_rw_pw_hard/controls'
    img_path = basepath + '/preprocessed/controls/numpy'

    # ==========================================
    # ==========================================
    num_images_to_load = len(os.listdir(seg_path))
    #name = patient.replace("seg_", "").replace("_.npy", "")
    i = 0
    for n, patient in enumerate(os.listdir(seg_path)):
        name = patient.replace("seg_", "").replace("_.npy", "")

        print("========================================================================")
        print('Loading subject ' + str(n+1) + ' out of ' + str(num_images_to_load) + '...')
        print('Patient\'s name: ' + name)
        print("========================================================================")

        # load the segmentation that was created with Nicolas's tool
        image = np.load(img_path + f'/{patient.replace("seg_", "")}')#np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy')
        segmented = np.load(seg_path + f'/{patient}')#np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/random_walker_prediction.npy')

        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])

        #Load the centerline coordinates for the given subject
        # centerline_coords = centerline_indexes[n]

        # print out the points
        for i in range(len(points)):
            print("Index {}:".format(str(i)) + str(points[i]))


    return 0


#====================================================================================
# CENTER LINE
#====================================================================================
def extract_slice_from_sitk_image(sitk_image, point, Z, X, new_size, fill_value=0):
    """
    Extract oblique slice from SimpleITK image. Efficient, because it rotates the grid and
    only samples the desired slice.

    """
    num_dim = sitk_image.GetDimension()

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())

    new_size = [int(el) for el in new_size]  # SimpleITK expects lists, not ndarrays
    point = [float(el) for el in point]

    rotation_center = sitk_image.TransformContinuousIndexToPhysicalPoint(point)

    X = X / np.linalg.norm(X)
    Z = Z / np.linalg.norm(Z)
    assert np.dot(X, Z) < 1e-12, 'the two input vectors are not perpendicular!'
    Y = np.cross(Z, X)

    orig_frame = np.array(orig_direction).reshape(num_dim, num_dim)
    new_frame = np.array([X, Y, Z])

    # important: when resampling images, the transform is used to map points from the output image space into the input image space
    rot_matrix = np.dot(orig_frame, np.linalg.pinv(new_frame))
    transform = sitk.AffineTransform(rot_matrix.flatten(), np.zeros(num_dim), rotation_center)

    phys_size = new_size * orig_spacing
    new_origin = rotation_center - phys_size / 2

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(orig_spacing)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetOutputOrigin(new_origin)
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resample_filter.SetTransform(transform)
    resample_filter.SetDefaultPixelValue(fill_value)

    resampled_sitk_image = resample_filter.Execute(sitk_image)
    
    
    """
    # Old method. Deprecated 
    
    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   transform,
                                                   sitk.sitkLinear,
                                                   new_origin,
                                                   orig_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)
    """
    return resampled_sitk_image



def interpolate_and_slice(image,
                          coords,
                          size):


    #coords are a bit confusing in order...
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    coords = np.array([z,y,x]).transpose([1,0])

    #convert the image to SITK (here let's use the intensity for now)
    sitk_image = sitk.GetImageFromArray(image[:,:,:])

    # spline parametrization
    params = [i / (size[2] - 1) for i in range(size[2])]
    tck, _ = interpolate.splprep(np.swapaxes(coords, 0, 1), k=3, s=200)

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
    return np.concatenate(slices, axis=2)


# ============================================
# Batch plotting helper functions
# ============================================

def tile_3d(X, rows, cols, every_x_time_step):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            img = X[i,:,:,j*every_x_time_step]
            tiling[
                    i*X.shape[1]:(i+1)*X.shape[1],
                    j*X.shape[2]:(j+1)*X.shape[2]] = img
    return tiling


def plot_batch_3d(X, channel, every_x_time_step, out_path):

    """
    This method creates a plot of a batch

    param: X - input of dimensions (batches, x, y, t,  channels)
    param: channel - which channel of the images should be plotted (0-3):(intensity,vx,vy,vz)
    param: every_x_time_step - for 1, all timesteps are plotted, for 2, every second timestep is plotted etc..
    param: out_path - path of the folder where the plots should be saved
    """

    X = np.stack(X)
    X = X[:,:,:,:,channel]

    rows = X.shape[0]
    cols = math.ceil(X.shape[3] // every_x_time_step)
    canvas = tile_3d(X, rows, cols, every_x_time_step)
    canvas = np.squeeze(canvas)

    plt.imsave(out_path, canvas, cmap='gray')



# ==========================================
# ==========================================


# ==========================================
# ==========================================
def create_center_lines(basepath):

    # ==========================================
    # ==========================================
    #num_images_to_load = idx_end + 1 - idx_start
    # For now Bern has 6 segmentations (no idea how to choose them)
    centerline_indexes = [
        [7, 43, 55, 81, 115, 130, 148],
        [11, 44, 87, 119, 131, 160, 190],
        [9, 30,  72, 89, 119, 150, 175],
        [2,  34, 55, 81, 120, 151, 170],
        [15, 34, 35, 74, 110, 150, 190],
        [5, 14, 34, 73, 90, 130, 180],
        ]
    """
    centerline_indexes = [
        [115, 81, 43, 7, 52, 120, 160],
        [87, 44, 11, 19, 89, 119, 131],
        [72, 30, 7, 49, 89, 119, 150],
        [94, 31, 0, 34, 81, 120, 151],
        [74, 34, 15, 35, 79, 110],
        [73, 14, 5, 34, 74, 119],
    ]
    """
    seg_path = basepath + '/segmenter_rw_pw_hard/controls'
    img_path = basepath + '/preprocessed/controls/numpy'
    

    i = 0
    for n, patient in enumerate(os.listdir(seg_path)):
        name = patient.replace("seg_", "").replace("_.npy", "")

        print("========================================================================")
        print('Loading subject ' + str(n+1) + ' out of ' + str(len(os.listdir(seg_path))) + '...')
        print('Patient\'s name: ' + name)
        print("========================================================================")

        # load the segmentation that was created with Nicolas's tool
        image = np.load(img_path + f'/{patient.replace("seg_", "")}')#np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy')
        segmented = np.load(seg_path + f'/{patient}')#np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/random_walker_prediction.npy')
        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])

        #Load the centerline coordinates for the given subject
        centerline_coords = centerline_indexes[n]
        
        

        # print out the points
        for i in range(len(points)):
            print("Index {}:".format(str(i)) + str(points[i]))

        #===========================================================================================
        # Parameters for the interpolation and creation of the files
        size = (30,30,256)
        coords = np.array(points[centerline_coords])

        # Create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, size)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)

        img_list = []
        img_list.append(straightened)

        # ================================
        # VISUALIZATIONS
        # ================================

        fig = plt.figure('Centerline')
        plt.imshow(image[:,:,15,3,0], cmap='gray')
        plt.scatter(points[:,1],points[:,0], s=2, c='red', marker='o')
        name = basepath + '/SubjectCenterlines/' + 'points_' + str(n) +'.png'
        fig.savefig(name)
        plt.close()


        num_slices = size[2]
        image_out  = img_list[0]
        # ================================
        # all slices of each time-index as png
        # ================================
        # %config InlineBackend.figure_format = "retina"
        figure2 = plt.figure(figsize=[120,120])
        for j in range(num_slices-1):
            plt.subplot(16, 16 , j+1)
            plt.imshow(image_out[:,:,j,3,0], cmap='gray')

        name = basepath + '/SubjectCenterlines/' + 'Straightened_' + str(n) +'.png'
        figure2.savefig(name)
        plt.close()

        figure2 = plt.figure(figsize=[120,120])
        for j in range(size[0]):
            plt.subplot(6, 6 , j+1)
            plt.imshow(image_out[:,j,:,3,0], cmap='gray')

        name = basepath + '/SubjectCenterlines/' + 'Straightened_SideView_' + str(n) +'.png'
        figure2.savefig(name)
        plt.close()

        #test plot batch function
        #(x,y,z,t,channel)
        for channel in range(4):
            name = basepath + '/SubjectCenterlines/' + 'Straightened_SideView_' + str(n) + 'channel_' + str(channel) +'.png'
            plot_batch_3d(image_out,channel,2,name)

        print("========================================================================\n\n")

    return 0


#====================================================================================
# CENTER LINE END ****
#====================================================================================





# ====================================================================================
# CROPPED AND STRAIGHTENED AORTA DATA Z-SLICES
#====================================================================================
def prepare_and_write_sliced_data(basepath,
                           filepath_output,
                           train_test,
                           stack_z):

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
    """
    #These are the manually selected points along the centerlines of the patients
    centerline_indexes = [
        [115, 81, 43, 7, 52, 120, 160],
        [87, 44, 11, 19, 89, 119, 131],
        [72, 30, 7, 49, 89, 119, 150],
        [94, 31, 0, 34, 81, 120, 151],
        [74, 34, 15, 35, 79, 110],
        [73, 14, 5, 34, 74, 119],
        [94, 54, 0, 55, 121, 151],
        [100, 52, 12, 63, 112, 134],
        [95, 55, 15, 40, 94, 118, 157],
        [105, 70, 16, 32, 93, 129, 162],
        [78, 34, 21, 35, 104, 134, 164],
        [74, 34, 6, 35, 100, 120, 153],
        [109, 71, 9, 26, 92, 131, 142, 174],
        [88, 21, 15, 66, 111, 141, 167]
    ]
    """

    # ==========================================
    # ==========================================
    seg_path = basepath + '/segmenter_rw_pw_hard/controls'
    img_path = basepath + '/preprocessed/controls/numpy'
    #num_images_to_load = idx_end + 1 - idx_start
    if train_test == 'train':
        patients = os.listdir(seg_path)[:4]
        num_images_to_load = len(patients)
    elif train_test == 'validation':
        patients = os.listdir(seg_path)[4:]
        num_images_to_load = len(patients)


    if stack_z == True:
        # ==========================================
        # we will stack all images along their z-axis
        # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
        # ==========================================
        images_dataset_shape = [end_shape[2]*num_images_to_load,
                                end_shape[0],
                                end_shape[1],
                                end_shape[3],
                                end_shape[4]]
    else:
        # ==========================================
        # If we are not stacking along z (the centerline of the cropped aorta),
        # we are stacking along y (so shape[1])
        # ==========================================
        images_dataset_shape = [end_shape[1]*num_images_to_load,
                                end_shape[0],
                                end_shape[2],
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
    if stack_z == True:
        dataset['sliced_images_%s' % train_test] = hdf5_file.create_dataset("sliced_images_%s" % train_test, images_dataset_shape, dtype='float32')
    else:
        dataset['straightened_images_%s' % train_test] = hdf5_file.create_dataset("straightened_images_%s" % train_test, images_dataset_shape, dtype='uint8')

    i = 0
    for patient in patients: 

        #print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        print('loading subject ' + str(i) + ' out of ' + str(num_images_to_load) + '...')

        # load the segmentation that was created with Nicolas's tool
        image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
        #image = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy')
        #segmented = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/random_walker_prediction.npy')
        segmented = np.load(os.path.join(seg_path, patient))
        
        # UPDATE TEMPORARY
        image = normalize_image(image)

        # The following lines were already commented out in the original code
        # temp_images_intensity = image[:,:,:,:,0] * segmented
        # temp_images_vx = image[:,:,:,:,1] * segmented
        # temp_images_vy = image[:,:,:,:,2] * segmented
        # temp_images_vz = image[:,:,:,:,3] * segmented

        # recombine the images
        # image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)


        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])

        # Limit to sectors where ascending aorta is located
        points = points[np.where(points[:,1]<60)]
        points = points[np.where(points[:,0]<100)]

        # Order the points in ascending order with x
        points = points[points[:,0].argsort()[::-1]]

        temp = []
        for index, element in enumerate(points[5:]):
            if (index%10)==0:
                temp.append(element)


        coords = np.array(temp)
        print(coords)

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)

        image_out  = straightened

        # make all images of the same shape
        print("Image shape before cropping and padding:" + str(image_out.shape))
        image_out = crop_or_pad_Bern_new(image_out, end_shape)
        print("Image shape after cropping and padding:" + str(image_out.shape))

        if stack_z == True:
            # move the z-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 2, 0)

            # add the image to the hdf5 file
            dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        else:
            # move the y-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 1, 0)

            print('After shuffling the axis' + str(image_out.shape))
            print(str(np.max(image_out)))

            # add the image to the hdf5 file
            dataset['straightened_images_%s' % train_test][i*end_shape[1]:(i+1)*end_shape[1], :, :, :, :] = image_out

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_cropped_data_sliced(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    dataset_filepath = basepath + '/sliced_images_from' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_sliced_data(basepath = basepath,
                               filepath_output = dataset_filepath,
                               
                               train_test = train_test,
                               stack_z = True)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

# ====================================================================================
# CROPPED AND STRAIGHTENED AORTA DATA Z-SLICES
#====================================================================================
def prepare_and_write_sliced_data_patient(basepath,
                           filepath_output,
                           train_test,
                           stack_z):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]

    #network_common_image_shape = [144, 112, None, 48, 4] # [x, y, t, num_channels]

    end_shape = [32, 32, 64, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)
    """
    #These are the manually selected points along the centerlines of the patients
    centerline_indexes = [
        [115, 81, 43, 7, 52, 120, 160],
        [87, 44, 11, 19, 89, 119, 131],
        [72, 30, 7, 49, 89, 119, 150],
        [94, 31, 0, 34, 81, 120, 151],
        [74, 34, 15, 35, 79, 110],
        [73, 14, 5, 34, 74, 119],
        [94, 54, 0, 55, 121, 151],
        [100, 52, 12, 63, 112, 134],
        [95, 55, 15, 40, 94, 118, 157],
        [105, 70, 16, 32, 93, 129, 162],
        [78, 34, 21, 35, 104, 134, 164],
        [74, 34, 6, 35, 100, 120, 153],
        [109, 71, 9, 26, 92, 131, 142, 174],
        [88, 21, 15, 66, 111, 141, 167]
    ]
    """

    # ==========================================
    # ==========================================
    

    image_path_base = "/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady"
    img_path = image_path_base + '/preprocessed/controls/numpy'

    patients = os.listdir(basepath)
#    patients = patients[:2]
    # The basepath is the model directory
    num_images_to_load = len(patients)


    if stack_z == True:
        # ==========================================
        # we will stack all images along their z-axis
        # --> the network will analyze (x,y,t) volumes, with z-samples being treated independently.
        # ==========================================
        images_dataset_shape = [end_shape[2]*num_images_to_load,
                                end_shape[0],
                                end_shape[1],
                                end_shape[3],
                                end_shape[4]]
    else:
        # ==========================================
        # If we are not stacking along z (the centerline of the cropped aorta),
        # we are stacking along y (so shape[1])
        # ==========================================
        images_dataset_shape = [end_shape[1]*num_images_to_load,
                                end_shape[0],
                                end_shape[2],
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
    if stack_z == True:
        dataset['sliced_images_%s' % train_test] = hdf5_file.create_dataset("sliced_images_%s" % train_test, images_dataset_shape, dtype='float32')
    else:
        dataset['straightened_images_%s' % train_test] = hdf5_file.create_dataset("straightened_images_%s" % train_test, images_dataset_shape, dtype='uint8')

    i = 0
    for patient in patients: 

        #print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        print('loading subject ' + str(i) + ' out of ' + str(num_images_to_load) + '...')

        # load the segmentation that was created with Nicolas's tool
        image = np.load(os.path.join(img_path, patient.replace("_seg_cnn", "")))
        #image = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy')
        #segmented = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/random_walker_prediction.npy')
        segmented = np.load(os.path.join(basepath, patient))
        
        # UPDATE TEMPORARY
        image = normalize_image(image)



        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])

        # Limit to sectors where ascending aorta is located
        points = points[np.where(points[:,1]<60)]
        points = points[np.where(points[:,0]<100)]

        # Order the points in ascending order with x
        points = points[points[:,0].argsort()[::-1]]

        temp = []
        for index, element in enumerate(points[5:]):
            if (index%10)==0:
                temp.append(element)


        coords = np.array(temp)
        print(coords)

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)

        image_out  = straightened

        # make all images of the same shape
        print("Image shape before cropping and padding:" + str(image_out.shape))
        #image_out = crop_or_pad_Bern_all_slices(image_out, network_common_image_shape)
        image_out = crop_or_pad_4dvol(image_out, end_shape)
        print("Image shape after cropping and padding:" + str(image_out.shape))

        if stack_z == True:
            # move the z-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 2, 0)

            # add the image to the hdf5 file
            dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        else:
            # move the y-axis to the front, as we want to stack the data along this axis
            image_out = np.moveaxis(image_out, 1, 0)

            print('After shuffling the axis' + str(image_out.shape))
            print(str(np.max(image_out)))

            # add the image to the hdf5 file
            dataset['straightened_images_%s' % train_test][i*end_shape[1]:(i+1)*end_shape[1], :, :, :, :] = image_out

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_cropped_data_sliced_patient(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    dataset_filepath = os.path.abspath(os.path.join(basepath, os.pardir)) + '/sliced_images_from' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_sliced_data_patient(basepath = basepath,
                               filepath_output = dataset_filepath,
                               
                               train_test = train_test,
                               stack_z = True)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

# ==========================================
# ==========================================
def load_cropped_data_straightened(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    dataset_filepath = basepath + '/straightened_images_from' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_sliced_data(basepath = basepath,
                               filepath_output = dataset_filepath,
                               train_test = train_test,
                               stack_z = False)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

# ====================================================================================
# *** CROPPED AND STRAIGHTENED AORTA DATA SLICED ****
#====================================================================================
# ====================================================================================
# MASKED DATA SLICED
#====================================================================================
def prepare_and_write_masked_data_sliced(basepath,
                           filepath_output,
                           idx_start,
                           idx_end,
                           train_test,
                           load_anomalous=False):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]
    common_label_shape = [144, 112, 32, 48] # [x, y, z, t]
    end_shape = [32, 32, 64, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    #num_images_to_load = idx_end + 1 - idx_start
    seg_path = basepath + '/segmenter_rw_pw_hard/controls'
    img_path = basepath + '/preprocessed/controls/numpy'
    #num_images_to_load = idx_end + 1 - idx_start
    if train_test == 'train':
        patients = os.listdir(seg_path)[:4]
        num_images_to_load = len(patients)
    elif train_test == 'validation':
        patients = os.listdir(seg_path)[4:]
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
    dataset['sliced_images_%s' % train_test] = hdf5_file.create_dataset("sliced_images_%s" % train_test, images_dataset_shape, dtype='float32')
    #dataset['labels_%s' % train_test] = hdf5_file.create_dataset("labels_%s" % train_test, labels_dataset_shape, dtype='uint8')
    
        
        
        
        
    i = 0
    for patient in patients: 
        
        #print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        print('loading subject ' + str(i) + ' out of ' + str(num_images_to_load) + '...')
        # load the segmentation that was created with Nicolas's tool
        if load_anomalous:
            image = np.load(basepath + '/' + 'anomalous_subject' + '/image.npy')
            segmented_original = np.load(basepath + '/' + 'anomalous_subject' + '/random_walker_prediction.npy')
        else:
            #image = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy')
            image = np.load(os.path.join(img_path, patient.replace("seg_", "")))
            #segmented_original = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/random_walker_prediction.npy')
            segmented_original = np.load(os.path.join(seg_path, patient))
        

        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        segmented = dilation(segmented_original[:,:,:,7], cube(3))

        temp_for_stack = [segmented for i in range(time_steps)]
        segmented = np.stack(temp_for_stack, axis=3)

        # normalize image to -1 to 1
        image = normalize_image_new(image)

        temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented

        # recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)


        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented_original, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])

        print(points)

        # Limit to sectors where ascending aorta is located
        points = points[np.where(points[:,1]<60)]
        points = points[np.where(points[:,0]<100)]

        # Order the points in ascending order with x
        points = points[points[:,0].argsort()[::-1]]

        temp = []
        for index, element in enumerate(points[5:]):
            if (index%5)==0:
                temp.append(element)

        coords = np.array(temp)
        print(coords)

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)
        image_out = straightened

        # make all images of the same shape
        print("Image shape before cropping and padding:" + str(image_out.shape))
        image_out = crop_or_pad_Bern_new(image_out, end_shape)
        print("Image shape after cropping and padding:" + str(image_out.shape))

        # move the z-axis to the front, as we want to stack the data along this axis
        image_out = np.moveaxis(image_out, 2, 0)

        # add the image to the hdf5 file
        dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_masked_data_sliced(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False,
              load_anomalous=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    dataset_filepath = basepath + '/masked_sliced_images_from' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_masked_data_sliced(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                               idx_end = idx_end,
                               train_test = train_test,
                               load_anomalous= load_anomalous)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

def prepare_and_write_masked_data_sliced_patient(basepath,
                           filepath_output,
                           idx_start,
                           idx_end,
                           train_test,
                           load_anomalous=False):

    # ==========================================
    # Study the the variation in the sizes along various dimensions (using the function 'find_shapes'),
    # Using this knowledge, let us set common shapes for all subjects.
    # ==========================================
    # This shape must be the same in the file where all the training parameters are set!
    # ==========================================
    common_image_shape = [36, 36, 64, 48, 4] # [x, y, z, t, num_channels]
    common_label_shape = [144, 112, 32, 48] # [x, y, z, t]
    end_shape = [32, 32, 64, 48, 4]
    # for x and y axes, we can remove zeros from the sides such that the dimensions are divisible by 16
    # (not required, but this makes it nice while training CNNs)

    # ==========================================
    # ==========================================
    #num_images_to_load = idx_end + 1 - idx_start
    image_path_base = "/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady"
    img_path = image_path_base + '/preprocessed/controls/numpy'

    patients = os.listdir(basepath)
    #patients = patients[:2]
    # The basepath is the model directory
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
    dataset['sliced_images_%s' % train_test] = hdf5_file.create_dataset("sliced_images_%s" % train_test, images_dataset_shape, dtype='float32')
    #dataset['labels_%s' % train_test] = hdf5_file.create_dataset("labels_%s" % train_test, labels_dataset_shape, dtype='uint8')
    
        
        
        
        
    i = 0
    for patient in patients: 
        
        #print('loading subject ' + str(n-idx_start+1) + ' out of ' + str(num_images_to_load) + '...')
        print('loading subject ' + str(i) + ' out of ' + str(num_images_to_load) + '...')
        # load the segmentation that was created with Nicolas's tool
        if load_anomalous:
            image = np.load(basepath + '/' + 'anomalous_subject' + '/image.npy')
            segmented_original = np.load(basepath + '/' + 'anomalous_subject' + '/random_walker_prediction.npy')
        else:
            #image = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/image.npy')
            image = np.load(os.path.join(img_path, patient.replace("_seg_cnn", "")))
            #segmented_original = np.load(basepath + '/' + subjects_ordering.SUBJECT_DIRS[n] + '/random_walker_prediction.npy')
            segmented_original = np.load(os.path.join(basepath, patient))
        

        # Enlarge the segmentation slightly to be sure that there are no cutoffs of the aorta
        time_steps = segmented_original.shape[3]
        segmented = dilation(segmented_original[:,:,:,7], cube(3))

        temp_for_stack = [segmented for i in range(time_steps)]
        segmented = np.stack(temp_for_stack, axis=3)

        # normalize image to -1 to 1
        image = normalize_image_new(image)
        seg_shape = list(segmented.shape)
        seg_shape.append(image.shape[-1])
        image = crop_or_pad_Bern_slices(image, seg_shape)


        temp_images_intensity = image[:,:,:,:,0] * segmented # change these back if it works
        temp_images_vx = image[:,:,:,:,1] * segmented
        temp_images_vy = image[:,:,:,:,2] * segmented
        temp_images_vz = image[:,:,:,:,3] * segmented

        # recombine the images
        image = np.stack([temp_images_intensity,temp_images_vx,temp_images_vy,temp_images_vz], axis=4)


        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(segmented_original, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])

        print(points)

        # Limit to sectors where ascending aorta is located
        points = points[np.where(points[:,1]<60)]
        points = points[np.where(points[:,0]<100)]

        # Order the points in ascending order with x
        points = points[points[:,0].argsort()[::-1]]

        temp = []
        for index, element in enumerate(points[5:]):
            if (index%5)==0:
                temp.append(element)

        coords = np.array(temp)
        print(coords)

        #===========================================================================================
        # Parameters for the interpolation and creation of the files

        # We create Slices across time and channels in a double for loop
        temp_for_channel_stacking = []
        for channel in range(image.shape[4]):

            temp_for_time_stacking = []
            for t in range(image.shape[3]):
                straightened = interpolate_and_slice(image[:,:,:,t,channel], coords, common_image_shape)
                temp_for_time_stacking.append(straightened)

            channel_stacked = np.stack(temp_for_time_stacking, axis=-1)
            temp_for_channel_stacking.append(channel_stacked)

        straightened = np.stack(temp_for_channel_stacking, axis=-1)
        image_out = straightened

        # make all images of the same shape
        print("Image shape before cropping and padding:" + str(image_out.shape))
        image_out = crop_or_pad_Bern_new(image_out, end_shape)
        print("Image shape after cropping and padding:" + str(image_out.shape))

        # move the z-axis to the front, as we want to stack the data along this axis
        image_out = np.moveaxis(image_out, 2, 0)

        # add the image to the hdf5 file
        dataset['sliced_images_%s' % train_test][i*end_shape[2]:(i+1)*end_shape[2], :, :, :, :] = image_out

        # increment the index being used to write in the hdf5 datasets
        i = i + 1

    # ==========================================
    # close the hdf5 file
    # ==========================================
    hdf5_file.close()

    return 0

# ==========================================
# ==========================================
def load_masked_data_sliced_patient(basepath,
              idx_start,
              idx_end,
              train_test,
              force_overwrite=False,
              load_anomalous=False):

    # ==========================================
    # define file paths for images and labels
    # ==========================================
    dataset_filepath = os.path.abspath(os.path.join(basepath, os.pardir)) + '/masked_sliced_images_from' + str(idx_start) + '_to_' + str(idx_end) + '.hdf5'

    if not os.path.exists(dataset_filepath) or force_overwrite:
        print('This configuration has not yet been preprocessed.')
        print('Preprocessing now...')
        prepare_and_write_masked_data_sliced_patient(basepath = basepath,
                               filepath_output = dataset_filepath,
                               idx_start = idx_start,
                               idx_end = idx_end,
                               train_test = train_test,
                               load_anomalous= load_anomalous)
    else:
        print('Already preprocessed this configuration. Loading now...')

    return h5py.File(dataset_filepath, 'r')

# ====================================================================================
# *** MASKED SLICED DATA END ****
#====================================================================================


if __name__ == "__main__":

    model_path = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/logdir/inference_results/unet3d_da_0.0nchannels4_r1_loss_dice_cut_z_True_full_run_bern_full_fine_tuning_lr_1e-3/controls'
    load_cropped_data_sliced_data = load_cropped_data_sliced_patient(model_path, 0, 0, 'train', force_overwrite=True)
    load_masked_data_patient(model_path,0,0, 'train', force_overwrite=True)
    masked_cropped_sliced_data = load_masked_data_sliced_patient(model_path, 0, 0, 'train', force_overwrite=True)