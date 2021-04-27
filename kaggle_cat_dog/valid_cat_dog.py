import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as Data
import torchvision


def remove_files(path: str):
    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))


def get_valid():
    train_path = "./valid"
    X = []
    y = []
    file_names = os.listdir(train_path)
    for file_name in file_names:
        img = cv2.imread(os.path.join(train_path, file_name))
        img_size = (100, 100)
        img = cv2.warpAffine(img, np.float32([[img_size[0] / img.shape[1], 0, 0], [0, img_size[1] / img.shape[0], 0]]), img_size)
        img = img / 256.0
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img.astype(np.float32)
        X.append(img)
        y.append(0 if file_name.startswith("c") else 1)
        if len(X) % 2000 == 0:
            print("processing valid %d %d" % (len(X), len(os.listdir(train_path))))
    return X, y


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 3x100x100->16x50x50
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 16x50x50->32x25x25
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 32x25x25->64x12x12
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 64x12x12->128x6x6
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.mlp = nn.Linear(128*6*6, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp(x.reshape(x.shape[0], -1))
        return x


cnn = CNN()
valid_X, valid_y = get_valid()
valid_X = torch.tensor(valid_X)
valid_y = torch.tensor(valid_y)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)

# load model
ck = torch.load("./model/saver.pkl")
cnn.load_state_dict(ck['net'])

print('start to train')
prediction_in_valid = cnn(valid_X.to(device))
pred_y = torch.max(prediction_in_valid.cpu(), 1)[1].data.numpy()
accuracy = float((pred_y == valid_y.data.numpy()).astype(int).sum()) / float(valid_y.size(0))
print('acc = %.2f' % accuracy)

remove_files("./right_cls")
remove_files("./wrong_cls")
for i in range(pred_y.shape[0]):
    img = valid_X[i].cpu().data.numpy()
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    soft_pred = F.softmax(torch.tensor(prediction_in_valid[i].cpu())).data.numpy().tolist(), valid_y.data.numpy()[i]
    if pred_y[i] != valid_y.data.numpy()[i]:
        cv2.imwrite(os.path.join("./wrong_cls", "img %d %s.png" % (i, str(soft_pred))), img * 256)
    else:
        cv2.imwrite(os.path.join("./right_cls", "img %d %s.png" % (i, str(soft_pred))), img * 256)
