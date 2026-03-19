import torch.nn as nn
import torch.nn.functional as F
import torch

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, only_dice=False):
        super(DiceBCELoss, self).__init__()
        self.only_dice = only_dice

    def forward(self, inputs, targets, smooth=1e-6):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        if self.only_dice:
            return dice_loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss        
        return Dice_BCE

def iou_score(outputs, targets, threshold=0.5):
    outputs = (outputs > threshold).float()
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum() - intersection
    iou = intersection / union
    return iou.item()

def dice_coefficient(outputs, targets, threshold=0.5):
    outputs = (outputs > threshold).float()
    intersection = (outputs * targets).sum()
    dice = (2 * intersection) / (outputs.sum() + targets.sum())
    return dice.item()

def pixel_accuracy(output, target, threshold=0.5):
    """
    Compute pixel accuracy for binary segmentation.
    
    Args:
        output (torch.Tensor): Model output probabilities (B, 1, H, W) or (B, H, W)
        target (torch.Tensor): Ground truth binary masks (B, 1, H, W) or (B, H, W)
        threshold (float): Threshold for binarizing output probabilities (default: 0.5)
    
    Returns:
        float: Pixel accuracy score
    """
    # Ensure inputs are on the same device
    output = output.to(target.device)
    
    # Apply sigmoid if output is logits (not probabilities)
    if output.max() > 1.0 or output.min() < 0.0:
        output = torch.sigmoid(output)
    
    # Binarize output based on threshold
    output = (output > threshold).float()
    
    # Ensure target is binary (0 or 1)
    target = (target > 0.5).float()
    
    # Flatten tensors to compute accuracy
    correct = (output == target).float().sum()
    total = output.numel()
    
    # Compute accuracy
    accuracy = correct / total
    return accuracy.item()