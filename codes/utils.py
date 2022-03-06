import torch
import torchvision.transforms as transforms

from torchvision import models
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader, Dataset
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

def _clusterInBatch(x, y):
    """cluster and order features into same-class group"""
    batch_size = y.size()[0]

    with torch.no_grad():
        sorted_y, indices = torch.sort(y)
        sorted_x = torch.zeros_like(x)
        for idx, order in enumerate(indices):
            sorted_x[idx] = x[order]

        intervals = []
        ex = 0
        for idx, val in enumerate(sorted_y):
            if ex == val:
                continue
            intervals.append(idx)
            ex = val
        intervals.append(batch_size)

        x = sorted_x
        y = sorted_y
    return x, y, intervals


def _shuffleBatch(output, proj, intervals):
    """generate shuffled batches"""
    output_2 = torch.zeros_like(output)
    feat_2 = torch.zeros_like(proj)
    output_3 = torch.zeros_like(output)
    feat_3 = torch.zeros_like(proj)
    ex = 0

    for end in intervals:
        shuffle_indices = torch.randperm(end - ex) + ex
        shuffle_indices2 = torch.randperm(end - ex) + ex
        for idx in range(end - ex):
            output_2[idx + ex] = output[shuffle_indices[idx]]
            feat_2[idx + ex] = proj[shuffle_indices[idx]]
            output_3[idx + ex] = output[shuffle_indices2[idx]]
            feat_3[idx + ex] = proj[shuffle_indices2[idx]]
        ex = end
    return output_2, output_3, feat_2, feat_3


def _selfregLoss(
    output, feat, proj, intervals, c_scale, SelfReg_criterion=nn.MSELoss()
):
    output_2, output_3, feat_2, feat_3 = _shuffleBatch(output, proj, intervals)

    lam = np.random.beta(0.5, 0.5)

    # mixup
    output_3 = lam * output_2 + (1 - lam) * output_3
    feat_3 = lam * feat_2 + (1 - lam) * feat_3

    # regularization
    L_ind_logit = SelfReg_criterion(output, output_2)
    L_hdl_logit = SelfReg_criterion(output, output_3)
    L_ind_feat = 0.3 * SelfReg_criterion(feat, feat_2)
    L_hdl_feat = 0.3 * SelfReg_criterion(feat, feat_3)

    return c_scale * (
        lam * (L_ind_logit + L_ind_feat) + (1 - lam) * (L_hdl_logit + L_hdl_feat)
    )


def plotting(losses, accuracies, used_model, save_dir, is_pretrained, try_check):
    plt.figure(figsize=(8, 2))
    plt.subplots_adjust(wspace=0.2)

    plt.subplot(1, 2, 1)
    plt.title("$Loss$", fontsize=18)
    plt.plot(losses)
    plt.grid()
    plt.xlabel("$epochs$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(1, 2, 2)
    plt.title("$Accuracy$", fontsize=18)
    plt.plot(accuracies)
    plt.grid()
    plt.xlabel("$epochs$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if is_pretrained:
        plt.savefig(
            save_dir
            + "transfer_training_{}_{}.png".format(used_model, str(try_check).zfill(2)),
            dpi=300,
        )
    else:
        plt.savefig(
            save_dir
            + "scratch_training_{}_{}.png".format(used_model, str(try_check).zfill(2)),
            dpi=300,
        )
    plt.show()


def save_model(model, used_model, save_dir, is_pretrained, try_check):
    if is_pretrained:
        torch.save(
            model.state_dict(),
            save_dir + "transfer_{}_{}.pth".format(used_model, str(try_check).zfill(2)),
        )
    else:
        torch.save(
            model.state_dict(),
            save_dir + "scratch_{}_{}.pth".format(used_model, str(try_check).zfill(2)),
        )


def train(
    train_loaders,
    val_loaders,
    optimizer,
    model,
    lr_scheduler,
    swa_model,
    swa_scr,
    device,
    epochs,
    criterion,
    is_selfreg,
    is_idcl,
    SelfReg_criterion=nn.MSELoss(),
):
    val_losses = []
    val_accuracies = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_loss_val = 0
    avg_acc_val = 0

    SelfReg_criterion = nn.MSELoss()

    if is_idcl:
        epoch_1 = (epochs // 3) // 2
        epoch_2 = epochs // 3
    else:
        train_loader, val_loader = train_loaders, val_loaders

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0

        loss_val = 0
        acc_val = 0

        model.train()

        if is_idcl:
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

        for idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # cluster and order features into same-class group
            if is_selfreg:
                x, y, intervals = _clusterInBatch(x, y)
                output, feat = model.extract_features(x)
                proj = model.projection(feat)
            else:
                output = model(x)

            # loss
            cl_loss = criterion(output, y)
            if is_selfreg:
                selfreg = _selfregLoss(
                    output,
                    feat,
                    proj,
                    intervals,
                    c_scale=min(cl_loss.item(), 1.0),
                    SelfReg_criterion=SelfReg_criterion,
                )
                loss = cl_loss + selfreg
            else:
                loss = cl_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)

            epoch_accuracy += accuracy.item()
            epoch_loss += loss.item()

            batch += len(x)
            print(
                "Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.4f}".format(
                    epoch + 1,
                    batch,
                    len(train_loader.dataset),
                    100.0 * (batch / len(train_loader.dataset)),
                    100.0 * (accuracy.item() / len(x)),
                    loss.item(),
                )
            )

        # lr_scheduler.step()
        if epoch >= 24:
            swa_model.update_parameters(model)
        swa_scr.step()

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
                loss = criterion(output, y)
                _, preds = torch.max(output.data, 1)

                loss_val += loss.item()
                acc_val += torch.sum(preds == y.data)

        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size

        val_losses.append(avg_loss_val)
        val_accuracies.append(avg_acc_val)

        print("loss_val:", loss_val, val_set_size)
        print("acc_val:", acc_val.item(), val_set_size)

        print("Validation", "=" * 50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print()

    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    # print("Best validation acc: {:.4f}".format(best_acc))

    return swa_model, val_losses, val_accuracies


def test(
    device, model, criterion, test_loader, used_model, save_dir, try_check
):
    batch = 0
    test_accuracy = 0

    f = open(
        save_dir + "test_{}_{}_log.txt".format(used_model, str(try_check).zfill(2)), "w"
    )

    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            _, preds = torch.max(output.data, 1)
            accuracy = torch.sum(preds == y.data)

            test_accuracy += accuracy.item()
            batch += len(x)

            log = (
                "Test:  [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}".format(
                    batch,
                    len(test_loader.dataset),
                    100.0 * (idx + 1) / len(test_loader),
                    100.0 * (test_accuracy / batch),
                    loss.item(),
                )
            )
            print(log)
            f.write(log + "\n")
    f.close()

    return str(round(100.0 * (test_accuracy / batch), 2))


def save_route(test_idx, domains, dataset, save_name, used_model):
    dom = ""
    for i in range(4):
        if i == test_idx:
            continue
        dom += domains[i]
        dom += "+"
    save_dir = (
        os.path.join(used_model, dataset)
        + "/{}/".format(save_name)
        + dom[:-1]
        + "("
        + domains[test_idx]
        + ")/"
    )
    return save_dir


# +
def get_tf(augment=True):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if augment == False:
        train_tf = test_tf
    return train_tf, test_tf


# -

# save setting
def save_model_setting(settings, used_model, domains, dataset, save_name):
    text_file_name = "model_parameter_setting_info.txt"
    save_dir = os.path.join(used_model, dataset, save_name)

    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except:
        print("Error : Creating directory. " + save_dir)

    save_dir = os.path.join(save_dir, text_file_name)

    with open(save_dir, "w") as f:
        for key, value in settings.items():
            f.write(key + " : " + str(value) + "\n")
