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


class CatDogDataset(TensorDataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.filenames = os.listdir(root_path)

    # cat: 0 dog: 1
    def __getitem__(self, index):
        file_name = self.filenames[index]
        img = cv2.imread(os.path.join(self.root_path, file_name))
        # cv2.imshow("img", img)
        # print("label : ", file_name.find("cat") == -1, file_name)
        # cv2.waitKey(0)
        img_size = (112, 112)
        img = cv2.warpAffine(img, np.float32([[img_size[0] / img.shape[1], 0, 0], [0, img_size[1] / img.shape[0], 0]]),
                             img_size)
        img = torch.tensor(img, dtype=torch.float)
        img = img / 255.0
        img = img.permute(2, 0, 1)
        return img, torch.tensor(file_name.find("cat") == -1, dtype=torch.int64)

    def __len__(self):
        return len(self.filenames)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.deConv = nn.ConvTranspose2d(3, 3, (4, 4), stride=(2, 2), padding=(1, 1))
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.mlp = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.deConv(x)
        x = self.resnet(x)
        x = self.mlp(x)
        return x


# 用反卷积放大图片
if __name__ == "__main__":
    torch.manual_seed(0)
    cnn = CNN()
    device = torch.device("cuda:0")
    cnn = cnn.to(device)

    train_loader = DataLoader(CatDogDataset("./train2"), batch_size=20, shuffle=True)
    valid_loader = DataLoader(CatDogDataset("./valid"), batch_size=20, shuffle=True)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-4)
    writer_train = SummaryWriter(log_dir="./board/warpAffine_small_deconv_large/train")
    writer_valid = SummaryWriter(log_dir="./board/warpAffine_small_deconv_large/valid")
    print('start to train')
    idx_train = 0
    idx_valid = 0
    for epoch in range(7):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
