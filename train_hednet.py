import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse

from hednet import HedNet
#from hednet import EnsembleSkeletonNet

from training_functions import get_args, get_device, plot_losses
from trainer_ofda import Trainer

from datetime import date, datetime
import os
from utils.dice_bce_loss import DiceBCELoss
from utils.focal_loss import WeightedFocalLoss

from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet and SkeletonNet on images and target masks')
    parser.add_argument('-l', '--loss', type=str, choices=["bce","dice","focal"], default="bce", help='Loss function')
    parser.add_argument('-nf', '--n_features', type=int, choices=[16,32,64], default=32, help='UNet 1st convolution features')
    parser.add_argument('-lri', '--lr_i', type=int, choices=[2,3,4,5,6], default=3, help='Learning Rate 10**i')
    parser.add_argument('-wdi', '--wd_i', type=int, choices=[3,4,5,6], default=6, help='Weight decay')
    parser.add_argument('-m', '--max_epochs_without_improve', type=int, choices=range(11,20), default=15, help='Early stopping')
    # parser.add_argument('-ts', '--testing_subset', type=str, choices=["synthetic","ofda"], default="synthetic", help='testing subset')
    return parser.parse_args()

def build_model_output_path(args):
    # year, month, day = str(date.today()).split('-')
    now = datetime.now()
    path = "./checkpoints"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    checkpoint_name = f"model_hednet_{args.loss}_{args.n_features}_{args.lr_i}_{args.wd_i}_{now.strftime('%Y%m%d_%H%M%S')}.pth"
    return os.path.join(path, checkpoint_name)

if __name__ == '__main__':
    
    args = get_args()
    
    print("Training SkeletonNet")
    print("-"*30)
    print('Loss Function:', args.loss)
    print('Number of Features:', args.n_features)
    print('Learning Rate:', args.lr_i)
    print('Weight Decay:', args.wd_i)
    print('Maximum of epochs without improve:', args.max_epochs_without_improve)

    model_output_path = build_model_output_path(args)
    print('Model output path:', model_output_path)
        
    net = HedNet(n_channels=3, n_classes=1, bilinear=False, n_features=args.n_features)

    criterion = {'bce':torch.nn.BCEWithLogitsLoss(),'dice':DiceBCELoss(),'focal':WeightedFocalLoss(w_pos=16.683, w_neg=0.515, gamma=2.0)}
    
    device = get_device()
    net.to(device=device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=10**-(args.lr_i))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.96)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1, # reduce to 10%
        patience=10,
        verbose=True
    )
    
    # trainer = Trainer(net, device, test_ofda_subset=(args.testing_subset == "ofda"))
    # trainer = Trainer(net, device, test_ofda_subset=True)
    trainer = Trainer(net, device, pad=True)
    
    train_errors,val_errors = trainer.train_and_validate(epochs=500,
                                                            criterion=criterion[args.loss],
                                                            optimizer=optimizer,
                                                            scheduler=scheduler,
                                                            model_output_path=model_output_path,
                                                            max_epochs_without_improve=args.max_epochs_without_improve) # print min train and val losses

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    plot_losses(train_errors,val_errors,f"{args.loss.upper()} Loss",f"outputs/plot_hednet_{args.loss}_{timestamp}.png")
    
