#%%
import torch
import torch.nn.functional as F
#%%
## ======================================================================
## ======================================================================
def compute_dice(logits, labels, epsilon=1e-10):
    '''
    Computes the dice score between logits and labels
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :return: dice (per label, per image in the batch)
    '''

    
    prediction = F.softmax(logits, dim=1)
    intersection = torch.mul(prediction, labels)
    # labels = [8,2,144,112,48]

    reduction_axes = [2,3,4]        
    # compute area of intersection, area of GT, area of prediction (per image per label)
    tp = torch.sum(intersection, dim=reduction_axes)
    tp_plus_fp = torch.sum(prediction, dim=reduction_axes)
    tp_plus_fn = torch.sum(labels, dim=reduction_axes)

    # compute dice (per image per label)
    dice = 2 * tp / (tp_plus_fp + tp_plus_fn + epsilon)

    # =============================
    # if a certain label is missing in the GT of a certain image and also in the prediction,
    # dice[this_image,this_label] will be incorrectly computed as zero whereas it should be 1.
    # =============================

    # mean over all images in the batch and over all labels.
    mean_dice = torch.mean(dice)

    # mean over all images in the batch and over all foreground labels.
    mean_fg_dice = torch.mean(dice[:, 1:])

    return dice, mean_dice, mean_fg_dice

## ======================================================================
## ======================================================================
def dice_loss(logits, labels, weights=None):
    # logits = [8,2,144,112,48]
    # labels = [8,2,144,112,48]
    

    dice, mean_dice, mean_fg_dice = compute_dice(logits, labels)

    if weights is None:
        loss = 1 - mean_dice
        
    else:
        loss = 1 - dice
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weights = torch.Tensor([1, 47.6]).to(device)
        loss = torch.sum(torch.mean(loss * weights, dim = 0))


    return loss

## ======================================================================
## ======================================================================
def pixel_wise_cross_entropy_loss(logits, labels, weights=None):
    
    if weights is None:
    
        # Compute the cross-entropy loss
        Loss = torch.nn.CrossEntropyLoss()
        loss = Loss(logits, labels.long())
    else:
        Loss = torch.nn.CrossEntropyLoss(reduction='none')
        loss = torch.mean(Loss(logits, labels.long())*weights, dim = 0)
    
    return loss