import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import pandas as pd


def get_test():
    test_path = "./test"
    X = []
    for idx in range(1, len(os.listdir(test_path)) + 1):
        file_name = "%d.jpg" % idx
        img = cv2.imread(os.path.join(test_path, file_name))
        img_size = (100, 100)
        img = cv2.warpAffine(img, np.float32([[img_size[0] / img.shape[1], 0, 0], [0, img_size[1] / img.shape[0], 0]]), img_size)
        img = img / 256.0
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img.astype(np.float32)
        X.append(img)
        if len(X) % 2000 == 0:
            print("processing test %d %d" % (len(X), len(os.listdir(test_path))))
    return X


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1x100x100->16x50x50
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
test_X = get_test()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)

data_loader = DataLoader(TensorDataset(torch.tensor(test_X)), batch_size=20, shuffle=False)

print('start to change')

ck = torch.load("./model/saver.pkl")
cnn.load_state_dict(ck['net'])
print('start to test')

res = []
for i, (imgs) in enumerate(data_loader):
    imgs = imgs[0]
    imgs = imgs.to(device)
    prediction = cnn(imgs).cpu().data.numpy()

    for i in range(prediction.shape[0]):
        res.append(F.softmax(torch.tensor(prediction[i])).data.numpy().tolist()[1])

data = {
    'id': [u for u in range(1, len(res) + 1)],
    'label': res
}
df = pd.DataFrame(data, index=None)
df.to_csv(path_or_buf="./submission.csv", index=False)
