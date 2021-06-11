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
        img_size = (224, 224)
        img = cv2.warpAffine(img, np.float32([[img_size[0] / img.shape[1], 0, 0], [0, img_size[1] / img.shape[0], 0]]),
                             img_size)
        img = torch.tensor(img, dtype=torch.float)
        img = img / 255.0
        img = img.permute(2, 0, 1)
        return img, torch.tensor(file_name.find("cat") == -1, dtype=torch.int64)

    def __len__(self):
        return len(self.filenames)


class CNN(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cuda:0')):
        super(CNN, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.mlp = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    cnn = CNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(device)

    train_loader = DataLoader(CatDogDataset("./train"), batch_size=20, shuffle=True)
    valid_loader = DataLoader(CatDogDataset("./valid"), batch_size=20, shuffle=True)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-4)
    writer_train = SummaryWriter(log_dir="./board/CrossEntropyLoss/train")
    writer_valid = SummaryWriter(log_dir="./board/CrossEntropyLoss/valid")
    print('start to train')
    idx_train = 0
    idx_valid = 0
    for epoch in range(7):
        cnn.train()
        print('train')
        loss_all = []
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            prediction = cnn(imgs)
            loss = nn.CrossEntropyLoss()(prediction, labels)
            loss_all.append(loss)
            if i % 20 == 0:
                print('train i = %d, epoch = %d, loss = %f' % (i, epoch, loss))
                writer_train.add_scalar("train_loss", torch.mean(torch.tensor(loss_all)), idx_train)
                writer_train.flush()
                loss_all = []
                idx_train += 1

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
            if i % 5 == 0:
                print('valid i = %d, epoch = %d, loss = %f acc = %f' % (i, epoch, loss, acc_all[0]))
                writer_valid.add_scalar("valid_loss", torch.mean(torch.tensor(loss_all)), idx_valid)
                writer_valid.add_scalar("valid_acc", torch.mean(torch.tensor(acc_all)), idx_valid)
                acc_loss = []
                loss_all = []
                writer_valid.flush()
                idx_valid += 1
