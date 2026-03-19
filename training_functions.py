import torch
import argparse
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet and SkeletonNet on images and target masks')
    parser.add_argument('-a', '--architecture', type=str, choices=["unet","skeleton"], default="unet", help='Architecture UNet or SkeletonNet')
    parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Loss function')
    parser.add_argument('-nf', '--n_features', type=int, choices=[16,32,64], default=32, help='UNet 1st convolution features')
    parser.add_argument('-lri', '--lr_i', type=int, choices=[2,3,4,5,6], default=3, help='Learning Rate 10**i')
    parser.add_argument('-wdi', '--wd_i', type=int, choices=[3,4,5,6], default=6, help='Loss function')
    return parser.parse_args()

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device

def plot_losses(train_losses, val_losses, loss_name="Loss", save_name=""):
    if len(train_losses)!=len(val_losses):
        print("Error, both lists should have equal number of items")
    else:
        plt.figure()
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses, label='Validation loss')
        plt.xticks(range(len(train_losses))) 
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(loss_name)
        plt.tight_layout()
        epochs = list(range(0,len(train_losses)))
        plt.xticks(epochs[::10])
        if save_name:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()