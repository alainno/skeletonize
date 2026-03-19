import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

import torch
import torch.nn as nn
from torchvision.utils import make_grid
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from unet_skel import UNetSkeleton
from hednet import HedNet
# from hednet import EnsembleSkeletonNet

#from trainer import Trainer
from trainer_ofda import Trainer
from training_functions import get_device

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet and SkeletonNet')
    parser.add_argument('-a', '--architecture', type=str, choices=["unet","skeleton"], default="unet", help='Architecture UNet or SkeletonNet')
    parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Train Loss function')
    parser.add_argument('-nf', '--n_features', type=int, choices=[16,32,64], default=32, help='UNet 1st convolution features')
    parser.add_argument('-lri', '--lr_i', type=int, choices=[2,3,4,5,6], default=3, help='Learning Rate 10**i')
    parser.add_argument('-wdi', '--wd_i', type=int, choices=[3,4,5,6], default=6, help='Weight decay')
    parser.add_argument('-e', '--ensemble_type', type=str, choices=["inner","outer"], default="inner", help='Ensemble type')
    parser.add_argument('-ts', '--testing_subset', type=str, choices=["synthetic","ofda"], default="synthetic", help='testing subset')
    parser.add_argument('-y', '--year', type=str, choices=["2024"], default="2024", help='model creation year')
    parser.add_argument('-m', '--month', type=str, choices=["10"], default="", help='model creation month')
    parser.add_argument('-d', '--day', type=str, choices=["18"], default="", help='model creation day')
    parser.add_argument('-hour', '--hour', type=str, default="", help='model creation time')
    parser.add_argument('-pad', '--pad', type=bool, default=False, help='model image padding')
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    print('Test Hyperparameters:')
    print('-'*20)
    print('Architecture:', args.architecture)
    print('Train Loss:', args.loss)
    print('Testing Subset:', args.testing_subset)
    
    if args.architecture == 'unet':
        checkpoint = f'./checkpoints/model_unet_focal_32_3_6_20251106_172240.pth'
        net = UNetSkeleton(in_channels=3, out_channels=1, base_ch=args.n_features)
    elif args.architecture == 'skeleton':
        if args.ensemble_type == 'inner':
            net = HedNet(n_channels=3, n_classes=1, bilinear=False, n_features=args.n_features, use_cuda=0)
        if args.ensemble_type == 'outer':
            model1 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=0, n_features=32)
            model2 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4, n_features=32)
            net = EnsembleSkeletonNet(model1, model2)
        # checkpoint = f"./checkpoints/model3_snet_{args.ensemble_type}_{args.loss}.pth"
        checkpoint = f"./checkpoints/model_hednet_focal_32_3_6_20260313_171633.pth"

    print('Checkpoint:', checkpoint)
        
    device = get_device()
    print(f'Using {device} as device')
    net.to(device=device)
    
    trainer = Trainer(net, device, test_ofda_subset=(args.testing_subset == "ofda"), pad=args.pad)
    # trainer = Trainer(net, device, test_ofda_subset=True)
    
    trainer.net.load_state_dict(torch.load(checkpoint))
    
    # batch = next(iter(trainer.val_data_loader)) ## <--- validation
    # inputs, gt, output = trainer.test_output(batch=batch)
    inputs, gt, output = trainer.test_output(batch_size=4)

    ########
    # plot #
    ########

    plt.figure(figsize=(1*5,3*5))
    
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0, hspace=0.1)
    
    # show micrographs
    imagen = make_grid(inputs, nrow=1, padding=0, normalize=True)
    raw = imagen.permute(1,2,0)
    plt.subplot(gs[0]), plt.axis('off'), plt.title('(a)', y=-0.05), plt.imshow(raw)
    
    # show ground truth
    imagen = make_grid(gt, nrow=1, padding=0, normalize=True)
    gt = imagen.permute(1,2,0)
    plt.subplot(gs[1]), plt.axis('off'), plt.title('(b)', y=-0.05), plt.imshow(gt)
    
    # show prediction
    imagen = make_grid(output, nrow=1, padding=0, normalize=True)
    prediction = imagen.permute(1,2,0)
    plt.subplot(gs[2]), plt.axis('off'), plt.title('(c)', y=-0.05), plt.imshow(prediction)

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    # save image
    plt.savefig(f"./outputs/tmp-output-{timestamp}.png", bbox_inches='tight', pad_inches=0)
#    plt.show()