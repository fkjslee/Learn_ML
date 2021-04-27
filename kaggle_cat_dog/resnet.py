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
    train_path = "./train2"
    X = []
    y = []
    file_names = os.listdir(train_path)
    random.shuffle(file_names)
    for file_name in file_names:
        img = cv2.imread(os.path.join(train_path, file_name))
        img_size = (224, 224)
        img = cv2.warpAffine(img, np.float32([[img_size[0] / img.shape[1], 0, 0], [0, img_size[1] / img.shape[0], 0]]),
                             img_size)
        img = img / 255.0
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img.astype(np.float32)
        X.append(img)
        y.append(0 if file_name.startswith("c") else 1)
        if len(X) % 200 == 0:
            print("processing train %d %d" % (len(X), len(os.listdir(train_path))))
            break
    return X, y


def get_valid():
    train_path = "./valid"
    X = []
    y = []
    file_names = os.listdir(train_path)
    random.shuffle(file_names)
    for file_name in file_names:
        img = cv2.imread(os.path.join(train_path, file_name))
        img_size = (224, 224)
        img = cv2.warpAffine(img, np.float32([[img_size[0] / img.shape[1], 0, 0], [0, img_size[1] / img.shape[0], 0]]),
                             img_size)
        img = img / 255.0
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img.astype(np.float32)
        X.append(img)
        y.append(0 if file_name.startswith("c") else 1)
        if len(X) % 200 == 0:
            print("processing valid %d %d" % (len(X), len(os.listdir(train_path))))
            break
    return X, y


class CNN(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cuda:0')):
        super(CNN, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False
        self.mlp = nn.Linear(2048 * 7 * 7, 2)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        ck = x
        for i in range(8):
            if i == 4 or i == 5:
                for j in range(len(self.features[i])):
                    ck = self.features[i][j](ck)
                    print(self.features[i][j])
                    print(ck.shape)
            else:
                ck = self.features[i](ck)
                print(self.features[i])
                print(ck.shape)
        x = self.features(x)
        x = self.mlp(x.reshape(x.shape[0], -1))
        return x


if __name__ == "__main__":
    cnn = CNN()
    X, y = get_data()
    valid_X, valid_y = get_valid()
    valid_X = torch.tensor(valid_X)
    valid_y = torch.tensor(valid_y)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(device)

    data_loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=20, shuffle=True)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-4)
    print('start to train')
    for epoch in range(7):
        for i, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            prediction = cnn(imgs)
            loss = nn.CrossEntropyLoss()(prediction, labels)
            prediction_in_test = cnn(valid_X.to(device))
            pred_y = torch.max(prediction_in_test.cpu(), 1)[1].data.numpy()
            accuracy = float((pred_y == valid_y.data.numpy()).astype(int).sum()) / float(valid_y.size(0))
            if i % 20 == 0:
                print('i = %d, epoch = %d, loss = %.3f  acc = %.2f' % (i, epoch, loss.cpu().data.numpy(), accuracy))
                state = {'net': cnn.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'i': i}
                torch.save(state, "./model/saver.pkl")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
