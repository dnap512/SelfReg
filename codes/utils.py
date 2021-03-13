# -*- coding: utf-8 -*-
# + {}
import torch
import torchvision.transforms as transforms

from torchvision import models
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from torch import nn, optim
from collections import OrderedDict

from pathlib import Path
from PIL import Image

import os
import copy

import numpy as np
import matplotlib.pyplot as plt


from torch import nn
from torch.nn import functional as F


# -

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class DGImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,domain, root, transform=None, target_transform=None,
                 loader=Datasets.folder.default_loader, is_valid_file=None):
        super(DGImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.domain = domain
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, self.domain, target

# +
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

class ImagesDataset(Dataset):
    def __init__(self, folder, transform,target_val):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        self.target=torch.LongTensor([target_val])
        self.transform=transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img),self.target


# -

def plotting(losses, accuracies, used_model, save_dir, is_pretrained, try_check):
    plt.figure(figsize=(8,2))
    plt.subplots_adjust(wspace=0.2)

    plt.subplot(1,2,1)
    plt.title("$Loss$",fontsize = 18)
    plt.plot(losses)
    plt.grid()
    plt.xlabel("$epochs$", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    plt.subplot(1,2,2)
    plt.title("$Accuracy$", fontsize = 18)
    plt.plot(accuracies)
    plt.grid()
    plt.xlabel("$epochs$", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    if is_pretrained:
        plt.savefig(save_dir+'transfer_training_{}_{}.png'.format(used_model,str(try_check).zfill(2)), dpi=300)
    else:
        plt.savefig(save_dir+'scratch_training_{}_{}.png'.format(used_model,str(try_check).zfill(2)), dpi=300)
    plt.show()


def save_model(model, used_model, save_dir, is_pretrained, try_check):
    if is_pretrained:
        torch.save(model.state_dict(), 
                   save_dir+'transfer_{}_{}.pth'.format(used_model,str(try_check).zfill(2)))
    else:
        torch.save(model.state_dict(),
                   save_dir+'scratch_{}_{}.pth'.format(used_model,str(try_check).zfill(2)))


def classic_training(device, epochs, model,optimizer, 
                     criterion, train_loader, val_loader,
                     lr_scheduler, is_self_reg=False,
                     is_positive_pair=False,
                     SelfReg_criterion=nn.MSELoss(),
                     swa_model=0, swa_scr=0
                    ):
    
    val_losses = list()
    val_accuracies = list()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc=0.0
    avg_loss_val=0
    avg_acc_val=0
    
    val_set_size=len(val_loader.dataset)
    val_batches=len(val_loader)  
    
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0
        
        loss_val=0
        acc_val=0
        
        model.train(True)
        

        for idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            
            ########################
            ########## Self-reg
            ########################
            
            if is_self_reg or is_positive_pair:
                batch_size = y.size()[0]

                with torch.no_grad():
                    sorted_y, indices = torch.sort(y)
                    sorted_x = torch.zeros_like(x)
                    for idx, order in enumerate(indices):
                        sorted_x[idx] = x[order]

                    intervals = []
                    ex = 0
                    for idx, val in enumerate(sorted_y):
                        if ex==val:
                            continue
                        intervals.append(idx)
                        ex = val
                    intervals.append(batch_size)

                    x = sorted_x
                    y = sorted_y
                    
                    
                    
            ######################
            ######################
            optimizer.zero_grad()
            
            if is_self_reg:
                output, feat = model.extract_features(x)
                proj = model.projection(feat)
            elif is_positive_pair:
                output, feat = model.extract_features(x1)
                output2, feat2 = model.extract_features(x2)
            else:
                output = model(x)
            
            ########################
            ########## Self-reg
            ########################
            
            if is_self_reg:
                output_2 = torch.zeros_like(output)
                feat_2 = torch.zeros_like(proj)
                output_3 = torch.zeros_like(output)
                feat_3 = torch.zeros_like(proj)
                ex = 0
                for end in intervals:
                    shuffle_indices = torch.randperm(end-ex)+ex
                    shuffle_indices2 = torch.randperm(end-ex)+ex
                    for idx in range(end-ex):
                        output_2[idx+ex] = output[shuffle_indices[idx]]
                        feat_2[idx+ex] = proj[shuffle_indices[idx]]
                        output_3[idx+ex] = output[shuffle_indices2[idx]]
                        feat_3[idx+ex] = proj[shuffle_indices2[idx]]
                    ex = end
                    
                
                lam = np.random.beta(0.5,0.5)
                
                output_3 = lam*output_2 + (1-lam)*output_3
                feat_3 = lam*feat_2 + (1-lam)*feat_3
                
                SelfReg_loss = SelfReg_criterion(output, output_2)
                SelfReg_mixup = SelfReg_criterion(output, output_3)
                EFM_loss = 0.3 * SelfReg_criterion(feat, feat_2)
                EFM_mixup = 0.3 * SelfReg_criterion(feat, feat_3)
                
                cl_loss = criterion(output,y)
                c_scale = min(cl_loss.item(), 1.)
                loss = cl_loss + c_scale*(lam*(SelfReg_loss + EFM_loss)+(1-lam)*(SelfReg_mixup + EFM_mixup))
                
    
                
            else:
                cl_loss = criterion(output,y)
                loss = cl_loss
                
                
            loss.backward()
            optimizer.step()

            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)

            epoch_accuracy += accuracy.item()
            epoch_loss += loss.item()

            batch += len(x)
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                epoch+1, batch, len(train_loader.dataset),
                100.*(batch/len(train_loader.dataset)), 
                100.*(accuracy.item()/len(x)), loss.item()))
        
        if epoch>=24:
            swa_model.update_parameters(model)

        swa_scr.step()
#         lr_scheduler.step()
        torch.cuda.empty_cache()
         
        #validation
        model.eval()
        
        for i, data in enumerate(val_loader):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs) if epoch < 24 else swa_model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
    
            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)

        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size
        
        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)
        
        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)
        
        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print()
        
#         if avg_acc_val >= best_acc:
#             print("Best Weights Updated!")
#             best_acc = avg_acc_val
#             best_model_wts = copy.deepcopy(model.state_dict())
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    print("Best validation acc: {:.4f}".format(best_acc))    
    model.load_state_dict(best_model_wts)
            
    return swa_model, val_losses, val_accuracies


def classic_test(device, model,criterion, test_loader,used_model, save_dir, try_check):
    batch = 0
    test_accuracy = 0
    
    f = open(save_dir+'test_{}_{}_log.txt'.format(used_model,str(try_check).zfill(2)),'w')
    
    #for multi-class accuracy
    y_pred=[]
    y_true=[]

    model.eval()
    for idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output,y)

        
        _, preds = torch.max(output.data, 1)
        accuracy = torch.sum(preds == y.data)
        
        #for multi-class accuracy
        y_pred.append(preds)
        y_true.append(y.data)

        test_accuracy += accuracy.item()
        batch += len(x)

        log = 'Test:  [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
            batch, len(test_loader.dataset), 100.*(idx+1)/len(test_loader), 100.*(test_accuracy/batch), loss.item())
        print(log)
        f.write(log+'\n')
    f.close()
    
    return str(round(100.*(test_accuracy/batch),2)), y_pred, y_true


def IDCL_training(device, epochs, model,optimizer, criterion, 
                train_loaders, val_loaders,
                lr_scheduler, 
                swa_model, swa_scr,
                is_self_reg=False
               ):
    val_losses = list()
    val_accuracies = list()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_loss_val = 0
    avg_acc_val = 0
    

    epoch_1 = (epochs//3)//2
    epoch_2 = epochs//3
    
    SelfReg_criterion = nn.MSELoss() # squared_emd

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0
        
        loss_val=0
        acc_val=0
        
        model.train()
        
        if epoch < epoch_1:
            train_loader = train_loaders[0]
            val_loader = val_loaders[0]
        elif epoch_1 <= epoch and epoch < epoch_2:
            train_loader = train_loaders[1]
            val_loader = val_loaders[1]
        else:
            train_loader = train_loaders[2]
            val_loader = val_loaders[2]
        
        val_set_size = len(val_loader.dataset)
        val_batches = len(val_loader)  
        
        for idx, (x,  y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            ########################
            ########## Self-reg
            ########################
            
            if is_self_reg:
                batch_size = y.size()[0]

                with torch.no_grad():
                    sorted_y, indices = torch.sort(y)
                    sorted_x = torch.zeros_like(x)
                    for idx, order in enumerate(indices):
                        sorted_x[idx] = x[order]

                    intervals = []
                    ex = 0
                    for idx, val in enumerate(sorted_y):
                        if ex==val:
                            continue
                        intervals.append(idx)
                        ex = val
                    intervals.append(batch_size)

                    x = sorted_x
                    y = sorted_y
                    
                    
            ######################
            ######################
            optimizer.zero_grad()
            
            if is_self_reg:
                output, feat = model.extract_features(x)
                proj = model.projection(feat)
            else:
                output = model(x)
            
            ########################
            ########## Self-reg
            ########################
            
            
            if is_self_reg:
                
                output_2 = torch.zeros_like(output)
                feat_2 = torch.zeros_like(proj)
                output_3 = torch.zeros_like(output)
                feat_3 = torch.zeros_like(proj)
                ex = 0
                for end in intervals:
                    shuffle_indices = torch.randperm(end-ex)+ex
                    shuffle_indices2 = torch.randperm(end-ex)+ex
                    for idx in range(end-ex):
                        output_2[idx+ex] = output[shuffle_indices[idx]]
                        feat_2[idx+ex] = proj[shuffle_indices[idx]]
                        output_3[idx+ex] = output[shuffle_indices2[idx]]
                        feat_3[idx+ex] = proj[shuffle_indices2[idx]]
                    ex = end
                    
                
                lam = np.random.beta(0.5,0.5)
                
                output_3 = lam*output_2 + (1-lam)*output_3
                feat_3 = lam*feat_2 + (1-lam)*feat_3
                
                
        
                SelfReg_loss = SelfReg_criterion(output, output_2)
                SelfReg_mixup = SelfReg_criterion(output, output_3)
                EFM_loss = 0.3 * SelfReg_criterion(feat, feat_2)
                EFM_mixup = 0.3 * SelfReg_criterion(feat, feat_3)
                
        
                cl_loss = criterion(output,y)
                c_scale = min(cl_loss.item(), 1.)
                loss = cl_loss + c_scale*(lam*(SelfReg_loss + EFM_loss)+(1-lam)*(SelfReg_mixup + EFM_mixup))
            else:
                loss = cl_loss
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)

            epoch_accuracy += accuracy.item()
            epoch_loss += loss.item()

            batch += len(x)
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.4f}'.format(
                epoch+1, 
                batch, 
                len(train_loader.dataset),
                100.*(batch/len(train_loader.dataset)), 
                100.*(accuracy.item()/len(x)), 
                loss.item())
                )

        lr_scheduler.step()
        if epoch>=24:
            swa_model.update_parameters(model)

        swa_scr.step()
        torch.cuda.empty_cache()
         
        
        ###################################
        # Validation
        ###################################
        
        model.eval()
        
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                
                output = model(x)
                loss = criterion(output,y)
                _, preds = torch.max(output.data, 1)

                loss_val += loss.data
                acc_val += torch.sum(preds == y.data)
        
            torch.cuda.empty_cache()
        
               
        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size
        
        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)
        
        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)
        
        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        
    torch.optim.swa_utils.update_bn(train_loaders[2], swa_model, device=device)
    print("Best validation acc: {:.4f}".format(best_acc))    
            
    return swa_model, val_losses, val_accuracies


def save_route(test_idx, domains, dataset, save_name, used_model):
    dom = ''
    for i in range(4):
        if i==test_idx:
            continue
        dom += domains[i]
        dom += '+'
    save_dir = os.path.join(used_model,dataset)+'/{}/'.format(save_name)+dom[:-1]+'('+domains[test_idx]+')/'
    return save_dir


# +
def get_tf(augment=True):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4, .4, .4, .5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    if augment==False:
        train_tf = test_tf
    return train_tf, test_tf




# -

#모델 세팅 저장
def save_model_setting(settings,used_model,domains, dataset, save_name):
    text_file_name="model_parameter_setting_info.txt"
    save_dir = os.path.join(used_model,dataset,save_name)
        
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except:
        print('Error : Creating directory. '+ save_dir)
        
    save_dir=os.path.join(save_dir,text_file_name)
    
    with open(save_dir,"w") as f:
        for key,value in settings.items():
            f.write(key+" : "+str(value)+"\n") 

