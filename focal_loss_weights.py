import os
import torch
import numpy as np
from glob import glob
from PIL import Image

def compute_class_weights(mask_paths, positive_value=1):
    """
    Compute w_pos and w_neg for Weighted Focal Loss based on dataset masks.

    Args:
        mask_paths (list[str]): list of file paths to binary masks.
        positive_value (int or float): pixel value representing the positive class (default: 1).
    Returns:
        (w_pos, w_neg): tuple of floats
    """
    total_pos = 0
    total_neg = 0

    for path in mask_paths:
        mask = np.array(Image.open(path))
        # print(np.unique(mask))
        # break
        # Convert to binary if not already (handle grayscale masks)
        mask = (mask == positive_value).astype(np.uint8)
        total_pos += mask.sum()
        total_neg += mask.size - mask.sum()

    total = total_pos + total_neg
    w_pos = total / (2.0 * total_pos + 1e-8)
    w_neg = total / (2.0 * total_neg + 1e-8)
    return w_pos, w_neg

if __name__=="__main__":
    # Assuming your dataset has masks in 'data/masks/*.png'
    mask_files = glob("./datasets/ofda/train/masks/*.png")
    mask_files += glob("./datasets/simulated/train/masks/*.png")
    
    w_pos, w_neg = compute_class_weights(mask_files, 255)
    print(f"Computed weights -> w_pos: {w_pos:.3f}, w_neg: {w_neg:.3f}")
    
    # Example: initialize your loss
    # criterion = WeightedFocalLoss(w_pos=w_pos, w_neg=w_neg, gamma=2.0)
