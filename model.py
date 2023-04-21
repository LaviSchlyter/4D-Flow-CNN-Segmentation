import numpy as np
import torch
import losses
import logging

# ================================================================
# ================================================================
def inference(images,
              model_handle,
              training):
    '''
    Wrapper function to provide an interface to a model from the model_zoo inside of the model module. 
    '''

    return model_handle(images, training)

# ================================================================
# ================================================================
def predict(images,
            model_handle):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param volumes: An input volume tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''

    logits = model_handle(images, training = tf.constant(False, dtype=tf.bool))
    softmax = tf.nn.softmax(logits)
    prediction = tf.argmax(softmax, axis=-1)

    return logits, softmax, prediction

# ================================================================
# ================================================================
def loss(logits,
         labels,
         nlabels,
         loss_type,
         labels_as_1hot=False):
    '''
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'crossentropy'/'dice'/'crossentropy_and_dice'
    :return: The segmentation
    '''

    if labels_as_1hot is False:
        labels = tf.one_hot(labels, depth = nlabels, axis = -1)
    
    if loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
        
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
        
    else:
        raise ValueError('Unknown loss: %s' % loss_type)

    return segmentation_loss

# ================================================================
# ================================================================
def training_step(loss,
                  optimizer_handle,
                  learning_rate):
    '''
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :param optimizer_handle: A handle to one of the tf optimisers 
    :param learning_rate: Learning rate
    :return: The training operation
    '''

    train_op = optimizer_handle(learning_rate = learning_rate).minimize(loss)
    
    # =====================================
    # Add update_ops to the list of training ops.
    # This ensures that the batch norm non-trainable parameters (moving moments) are updated in each training iteration.
    # =====================================
    opt_memory_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([train_op, opt_memory_update_ops])

    return train_op

# ================================================================
# ================================================================
def evaluation(logits,
               labels,
               images,
               nlabels,
               loss_type):
    '''
    A function for evaluating the performance of the netwrok on a minibatch. This function returns the loss and the 
    current foreground Dice score, and also writes example segmentations and images to to tensorboard.
    :param logits: Output of network before softmax
    :param labels: Ground-truth label mask
    :param images: Input volume mini batch
    :param nlabels: Number of labels in the dataset
    :param loss_type: Which loss should be evaluated
    :return: The loss and the dice score of a minibatch
    '''

#    mask = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
#    mask_gt = labels

#    tf.summary.image('example_gt1', prepare_tensor_for_summary(mask_gt, mode='mask', idx=1, nlabels=nlabels))
#    tf.summary.image('example_pred1', prepare_tensor_for_summary(mask, mode='mask', idx=1, nlabels=nlabels))
#    tf.summary.image('example_zimg1', prepare_tensor_for_summary(volumes, mode='volume', idx=1))
#    
#    tf.summary.image('example_gt2', prepare_tensor_for_summary(mask_gt, mode='mask', idx=2, nlabels=nlabels))
#    tf.summary.image('example_pred2', prepare_tensor_for_summary(mask, mode='mask', idx=2, nlabels=nlabels))
#    tf.summary.image('example_zimg2', prepare_tensor_for_summary(volumes, mode='volume', idx=2))
#    
#    tf.summary.image('example_gt3', prepare_tensor_for_summary(mask_gt, mode='mask', idx=3, nlabels=nlabels))
#    tf.summary.image('example_pred3', prepare_tensor_for_summary(mask, mode='mask', idx=3, nlabels=nlabels))
#    tf.summary.image('example_zimg3', prepare_tensor_for_summary(volumes, mode='volume', idx=3))

    loss_, dice_ = evaluate_losses(logits,
                                   labels,
                                   nlabels,
                                   loss_type)

    return loss_, dice_

# ================================================================
# ================================================================
def evaluate_losses(logits,
                    labels,
                    nlabels,
                    loss_type):
    '''
    A function to compute various loss measures to compare the predicted and ground truth annotations
    '''
    
    labels = tf.one_hot(labels, depth = nlabels, axis = -1)
    
    loss_ = loss(logits,
                 labels,
                 nlabels = nlabels,
                 loss_type = loss_type,
                 labels_as_1hot = True)
    
    _, dice_ ,_ = losses.compute_dice(logits, labels)
    
    return loss_, dice_

# ================================================================
# ================================================================
def prepare_tensor_for_summary(vol,
                               mode,
                               idx = 0,
                               nlabels = None):
    '''
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param vol: Input volume or segmentation mask
    :param mode: Can be either 'volume' or 'mask'. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''
    print('volume.get_shape.ndims = ' +str(vol.get_shape().ndims))
    print('volume.get_shape = ' +str(vol.get_shape()))
    if mode == 'mask':

        if vol.get_shape().ndims == 3:
            V = tf.slice(vol, (idx, 0, 0), (1, -1, -1))
        elif vol.get_shape().ndims == 4:
            V = tf.slice(vol, (idx, 0, 0, 10), (1, -1, -1, 1))
        elif vol.get_shape().ndims == 5:
            V = tf.slice(vol, (idx, 0, 0, 10, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (vol.get_shape().ndims))

    elif mode == 'volume':

        if vol.get_shape().ndims == 3:
            V = tf.slice(vol, (idx, 0, 0), (1, -1, -1))
        elif vol.get_shape().ndims == 4:
            V = tf.slice(vol, (idx, 0, 0, 0), (1, -1, -1, 1))
        elif vol.get_shape().ndims == 5:
            V = tf.slice(vol, (idx, 0, 0, 10, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (vol.get_shape().ndims))

    else:
        raise ValueError('Unknown mode: %s. Must be volume or mask' % mode)

    if mode=='volume' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8)

    vol_w = tf.shape(vol)[1]
    vol_h = tf.shape(vol)[2]

    V = tf.reshape(V, tf.stack((-1, vol_w, vol_h, 1)))
    return V