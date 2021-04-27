import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as Data
import torchvision


def get_data():
    train_path = "./train"
    X = []
    y = []
    file_names = os.listdir(train_path)
    random.shuffle(file_names)
    for file_name in file_names:
        img = cv2.imread(os.path.join(train_path, file_name), 0)
        img_size = (100, 100)
        img = cv2.warpAffine(img, np.float32([[img_size[0] / img.shape[1], 0, 0], [0, img_size[1] / img.shape[0], 0]]), img_size)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = img / 256.0
        X.append(img)
        y.append(1 if file_name.startswith("c") else 0)
        if len(X) % 2000 == 0:
            print("processing train %d %d" % (len(X), len(os.listdir(train_path))))
    return X, y


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1x100x100->16x50x50
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
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
        self.mlp = nn.Linear(128*6*6, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp(x.reshape(x.shape[0], -1))
        return x


cnn = CNN()
X, y = get_data()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)

data_loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=20, shuffle=True)

print('start to change')

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
ck = torch.load("./model/saver.pkl")
cnn.load_state_dict(ck['net'])
optimizer.load_state_dict(ck['optim'])
print('start to train')
for epoch in range(7):
    for i, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        prediction = cnn(imgs)
        loss = nn.CrossEntropyLoss()(prediction, labels)

        pred_y = torch.max(prediction.cpu(), 1)[1].data.numpy()
        labels = labels.cpu()
        accuracy = float((pred_y == labels.data.numpy()).astype(int).sum()) / float(labels.size(0))
        if i % 20 == 0:
            print('i = %d, epoch = %d, loss = %.3f  acc = %.2f' % (i, epoch, loss.cpu().data.numpy(), accuracy))
            state = {'net': cnn.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'i':i}
            torch.save(state, "./model/saver.pkl")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


