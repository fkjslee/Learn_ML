import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as Data
import torchvision
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import timm
from vit_pytorch import ViT

from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue,
    RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return Compose([
        RandomResizedCrop(400, 400),
        # Transpose(p=0.5),
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        # ShiftScaleRotate(p=0.5),
        # HueSaturationValue(),
        # RandomBrightnessContrast(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        # CoarseDropout(),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms():
    return Compose([
        RandomResizedCrop(400, 400),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


class CatDogDataset(TensorDataset):
    def __init__(self, root_path, transform: Compose):
        super().__init__()
        self.root_path = root_path
        self.filenames = os.listdir(root_path)
        self.transform = transform

    # cat: 0 dog: 1
    def __getitem__(self, index):
        file_name = self.filenames[index]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root_path, file_name)), cv2.COLOR_BGR2RGB)
        img_size = (400, 400)
        img = cv2.warpAffine(img, np.float32([[img_size[0] / img.shape[1], 0, 0], [0, img_size[1] / img.shape[0], 0]]),
                             img_size)
        img = self.transform(image=img)['image']
        return img, torch.tensor(file_name.find("cat") == -1, dtype=torch.int64)

    def __len__(self):
        return len(self.filenames)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.vit = ViT(image_size=400, patch_size=40, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048)

    def forward(self, x):
        return self.vit(x)


# 主要结果
if __name__ == "__main__":
    torch.manual_seed(0)
    cnn = CNN()
    device = torch.device("cuda:0")
    cnn = cnn.to(device)
    # cnn.load_state_dict(torch.load("net.pkl"))

    train_loader = DataLoader(CatDogDataset("./train", get_train_transforms()), batch_size=5, shuffle=True)
    valid_loader = DataLoader(CatDogDataset("./valid", get_valid_transforms()), batch_size=5, shuffle=True)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-4)
    writer_train = SummaryWriter(log_dir="./board/vit/train")
    writer_valid = SummaryWriter(log_dir="./board/vit/valid")
    print('start to train')
    idx_train = 0
    idx_valid = 0
    for epoch in range(100000000):
        with open("./stop_file.txt", "r") as file:
            if file.readlines()[0].strip() != "0":
                break
        cnn.train()
        print('train')
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            prediction = cnn(imgs)
            loss = nn.CrossEntropyLoss()(prediction, labels)
            if i % 20 == 0:
                print('train i = %d, epoch = %d, loss = %f' % (i, epoch, loss))
                writer_train.add_scalar("train_loss", loss, idx_train)
                writer_train.flush()
                idx_train += 1
                torch.save(cnn.state_dict(), "net.pkl")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del i, imgs, labels, prediction, loss

        cnn.eval()
        print('validation')
        loss_all = []
        acc_all = []
        for i, (imgs, labels) in enumerate(valid_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            prediction = cnn(imgs)
            loss = nn.CrossEntropyLoss()(prediction, labels)
            loss_all.append(loss.clone().detach())
            y = labels
            x = torch.topk(prediction, 1, 1).indices.reshape(-1)
            acc_all.append(torch.sum(x == y) / x.shape[0])
        print('valid i = %d, epoch = %d, loss = %f acc = %f' % (i, epoch, loss, acc_all[0]))
        writer_valid.add_scalar("valid_loss", torch.mean(torch.tensor(loss_all)), idx_valid)
        writer_valid.add_scalar("valid_acc", torch.mean(torch.tensor(acc_all)), idx_valid)
        writer_valid.flush()
        idx_valid += 1
