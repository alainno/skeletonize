import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.transforms.v2 as transforms
# from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import argparse
import time

from unet import UNet
# from hednet import HedNet
from utils.dataset_aug import OfdaDataset
# from utils.dataset import BasicDataset
from utils.dice_bce_loss import DiceBCELoss, dice_coefficient, pixel_accuracy, iou_score

from training_functions import get_device
import random
# import matplotlib.pyplot as plt

def show(name, img):
    #npimg = img.numpy()
    #plt.figure()
    #plt.imsave('tmp-test-tfs-'+name+'.png', np.transpose(npimg, (1,2,0)))
    # plt.imsave('tmp-test-tfs-'+name+'.png', img.permute(1,2,0), cmap='gray')
    pass

class Trainer:
    def __init__(self, net, device, test_ofda_subset=False, pad=False):
        self.net = net
        self.device = device
        self.pad = pad

        img_path = "./datasets/ofda/train/images/"
        gt_path = "./datasets/ofda/train/masks/"
        
        geometric_augs = [
            transforms.CenterCrop(size=(189,189)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(90,90)),
            transforms.RandomRotation(degrees=(180,180)),
            transforms.RandomRotation(degrees=(270,270))
        ]

        if self.pad:
            geometric_augs += [transforms.Pad((1,1,2,2))]            

        color_augs = [
            #transforms.RandomInvert(1),
            transforms.GaussianBlur(kernel_size=9,sigma=(0.1,5.0)),
            #transforms.GaussianBlur(kernel_size=3,sigma=(0.1,5.0)),
            transforms.RandomAutocontrast(),
            transforms.RandomPosterize(bits=4)
        ]
        
        #transform_input = Trainer.make_tfs(geometric_augs + [transforms.RandomInvert(1)] + color_augs, normalize=True, mean=[0.726, 0.726, 0.726], std=[0.201, 0.201, 0.201])
        transform_input = Trainer.make_tfs(geometric_augs + [transforms.RandomInvert(1)] + color_augs, normalize=True, mean=[0.726, 0.726, 0.726], std=[0.201, 0.201, 0.201])
        transform_target = Trainer.make_tfs(geometric_augs)

        self.dataset = OfdaDataset(imgs_dir = img_path, masks_dir = gt_path, transforms=[transform_input, transform_target], mask_h5=False)
    
        syn_img_path = "./datasets/simulated/train/images/"
        syn_gt_path = "./datasets/simulated/train/masks/"
        
        transform_input = Trainer.make_tfs(geometric_augs + color_augs, normalize=True, mean=[0.190, 0.190, 0.190],std=[0.283, 0.283, 0.283])

        self.syn_dataset = OfdaDataset(imgs_dir = syn_img_path, masks_dir = syn_gt_path, transforms=[transform_input, transform_target], mask_h5=False)
        
        self.fused_dataset = torch.utils.data.ConcatDataset([self.syn_dataset, self.dataset])

        # r = random.randint(0, len(self.fused_dataset))
        # print(self.fused_dataset[r]['image'].shape)
        # print(self.fused_dataset[r]['mask'].shape)
        # print(self.fused_dataset[r]['image'].dtype)
        # print(self.fused_dataset[r]['mask'].dtype)
        # print(self.fused_dataset[r]['path'])
        
        val_percent = 0.2
        batch_size = 4

        self.n_val = int(len(self.fused_dataset) * val_percent)
        self.n_train = len(self.fused_dataset) - self.n_val
        self.train, self.val = random_split(self.fused_dataset, [self.n_train, self.n_val])

        self.train_data_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
        self.val_data_loader = DataLoader(self.val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

        # self.train_data_loader = self.__get_train_data_loader()
        # self.val_data_loader = self.__get_val_data_loader()
        self.test_ofda_subset = test_ofda_subset
        
        self.__init_test_dataset(batch_size=batch_size)

    @staticmethod
    def make_tfs(augs, normalize=False, mean=[.5,.5,.5], std=[.5,.5,.5]):
        if normalize:
            return transforms.Compose(augs + [transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])
        return transforms.Compose(augs + [transforms.ToTensor()])

    
    def __init_test_dataset(self, batch_size=2):
        #mean=[0.173, 0.173, 0.173] # synthetic overlapped
        #std=[0.289, 0.289, 0.289] # syntehtics overlapped
        mean=[0.190, 0.190, 0.190]
        std=[0.283, 0.283, 0.283]
        invert = 0
        if self.test_ofda_subset:
            test_img_path = "./datasets/ofda/test/images/"
            test_gt_path = "./datasets/ofda/test/masks/"
            invert = 1
            mean=[0.726, 0.726, 0.726]
            std=[0.201, 0.201, 0.201]
        else:
            test_img_path = "./datasets/simulated/test/images/"
            test_gt_path = "./datasets/simulated/test/masks/"

        trans_input = transforms.Compose([
            transforms.CenterCrop(size=(189,189)),
            transforms.RandomInvert(invert),#invert for obtain black background and white fibers
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        
        trans_target = transforms.Compose([
            transforms.CenterCrop(size=(189,189)),
            transforms.ToTensor()
        ])

        if self.pad:
            trans_input.transforms.append(transforms.Pad((1,1,2,2)))
            trans_target.transforms.append(transforms.Pad((1,1,2,2)))

        #test_dataset = BasicDataset(imgs_dir = test_img_path, masks_dir = test_gt_path, transforms=trans, mask_h5=True)
        self.test_dataset = OfdaDataset(imgs_dir = test_img_path, masks_dir = test_gt_path, transforms=[trans_input,trans_target], mask_h5=False)
        
        #self.test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        #self.test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)


    def __train(self, epoch):
        ''' Método de entrenamiento del modelo para una época'''
        self.net.train()
        epoch_loss = 0

        with tqdm(total=self.n_train, desc=f'Train Epoch {epoch+1}/{self.epochs}') as pbar:
            for batch in self.train_data_loader:
                input,ground_truth = batch['image'],batch['mask']
                input = input.to(device=self.device, dtype=torch.float32)
                ground_truth = ground_truth.to(device=self.device, dtype=torch.float32)

                #print(ground_truth.shape)
                
                output = self.net(input)
                
                #print(output.shape)

                loss = self.criterion(output, ground_truth)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(input.shape[0])
        
        return epoch_loss / len(self.train_data_loader)

    def __validate(self, epoch):
        ''' Método de validación del modelo'''
        self.net.eval()
        epoch_loss = 0

        with tqdm(total=self.n_val, desc=f'Val Epoch {epoch+1}/{self.epochs}') as pbar:
            for batch in self.val_data_loader:
                input,ground_truth = batch['image'],batch['mask']
                input = input.to(device=self.device, dtype=torch.float32)
                ground_truth = ground_truth.to(device=self.device, dtype=torch.float32)

                with torch.no_grad():
                    output = self.net(input)
                    loss = self.criterion(output, ground_truth)

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)':loss.item()})
                pbar.update(input.shape[0]) # samples in batch

        return epoch_loss / len(self.val_data_loader)

    def train_and_validate(self, epochs, criterion, optimizer, scheduler, model_output_path, max_epochs_without_improve=10):
        ''' Entrenar y validar modelo por varias épocas'''
        # train_data_loader = self.get_train_data_loader()
        # val_data_loader = self.get_val_data_loader()

        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        min_val_loss = np.Inf
        epochs_without_improve = 0
        max_epochs_without_improve = max_epochs_without_improve
        train_losses, val_losses = [],[]

        for epoch in range(epochs):
            #print(f"Epoch {epoch+1} de {epochs}")
            train_loss = self.__train(epoch)
            train_losses.append(train_loss)

            val_loss = self.__validate(epoch)
            val_losses.append(val_loss)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step() # para StepLR cada step_size

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_without_improve = 0
                torch.save(self.net.state_dict(), model_output_path)
            else:
                epochs_without_improve += 1
                if epochs_without_improve > max_epochs_without_improve:
                    print("Early stopping!")
                    break
        print("Min Training Loss:", min(train_losses))
        print("Min Validation Loss:", min(val_losses))
        return train_losses,val_losses
        
    def test(self, batch_size=4, printlog=False):
        if printlog:
            print("Iniciando el testing...")

        self.__init_test_dataset(batch_size=batch_size)

        if printlog:
            print(f'{len(self.test_data_loader)} test batches of {batch_size} samples loaded')

        # criterion = nn.BCEWithLogitsLoss()
        # criterion2 = DiceBCELoss(only_dice=True)

        self.net.eval()

        total_acc, total_dice, total_iou = 0, 0, 0
        
        total_time = 0
        
        for batch in self.test_data_loader:
            input, groundtruth = batch['image'], batch['mask']
            input = input.to(device=self.device, dtype=torch.float32)
            groundtruth = groundtruth.to(device=self.device, dtype=torch.float32)
            
            if printlog:
                # print(torch.unique(groundtruth[0]))
                print(f"path:{batch['path']}")

            with torch.no_grad():
                start_time = time.time()
                output = self.net(input)
                total_time += (time.time() - start_time)

                #print(output.shape)
                # loss = criterion(output, groundtruth)
                # loss2 = criterion2(output, groundtruth)
                pixel_acc = pixel_accuracy(output, groundtruth)
                dice_score = dice_coefficient(output, groundtruth)
                iou = iou_score(output, groundtruth)

            if printlog:
                print(f"batch pixel acc: {pixel_acc}")
                print(f"batch dice score: {dice_score}")
                print(f"batch iou score: {iou}")

            # if printlog:
                # print(f'batch bce: {loss.item()}, batch dice: {loss2.item()}')
                # print(f'batch bce: {loss.item()}, batch dice: {dice_score}')
                
            # test_loss += loss.item()
            # test_loss2 += loss2.item()
            total_acc += pixel_acc
            total_dice += dice_score
            total_iou += iou
        
        if printlog:
            print(f'* Execution time: {total_time}')
            print(f'* Time per image: {total_time/len(self.test_data_loader)}')
            print(f'* Images per second: {len(self.test_data_loader)/total_time}')
            # print('* len self test data loader:', len(self.test_data_loader))
            
            
        avg_acc = total_acc / len(self.test_data_loader)
        avg_dice = total_dice / len(self.test_data_loader)
        avg_iou = total_iou / len(self.test_data_loader)
        if printlog:
            print('* Pixel accuracy avg:', avg_acc)
            print('* Dice coefficient avg:', avg_dice)
            print('* iou coefficient avg:', avg_iou)
        return avg_acc, avg_dice, avg_iou

    def test_output(self, batch_size=4, printlog=False, batch=None):
        if printlog:
            print("Iniciando el testing...")

        self.net.eval()
        
        if batch is None:
            self.__init_test_dataset(batch_size=batch_size)
            batch = next(iter(self.test_data_loader))
            if printlog:
                print(f'{len(self.test_data_loader)} test batches of {batch_size} samples loaded')

        inputs, groundtruth = batch['image'], batch['mask']
        inputs = inputs.to(device=self.device, dtype=torch.float32)
        #groundtruth = groundtruth.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(inputs)
            output = F.sigmoid(output)
            binary_output = (output>0.5).float()

        return inputs.cpu(), groundtruth.cpu(), binary_output.cpu()
    
    
if __name__ == '__main__':
    
    net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=32)
    
    device = get_device()
    net.to(device=device)
    
    trainer = Trainer(net, device)

    print('total dataset items:', len(trainer.dataset))
    
    imgs = [trainer.dataset[i] for i in range(6)]
    
    show('imgs', torchvision.utils.make_grid(torch.stack([img['image'] for img in imgs])))
    show('masks', torchvision.utils.make_grid(torch.stack([img['mask'] for img in imgs])))
    
    print(imgs[0]['image'].max())
    print(imgs[0]['image'].min())
    