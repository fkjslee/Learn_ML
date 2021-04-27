import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
def get_data(path):
    train_pd = pd.read_csv(path)
    labels = train_pd.values[:, 0]
    imgs = train_pd.values[:, 1]
    X = []
    for i in range(imgs.shape[0]):
        img = np.fromstring(imgs[i], dtype=np.float32, sep=' ').reshape((48, 48))
        img = img / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        X.append(img)
    return np.float32(X), np.int64(labels)


def plot_heatmap(pred, label):
    import seaborn as sn
    import matplotlib.pyplot as plt
    import pandas as pd
    mat_range = torch.max(torch.max(torch.from_numpy(pred)), torch.max(torch.from_numpy(label))).data.numpy() + 1
    conf_matrix = torch.zeros(mat_range, mat_range, dtype=torch.int64)
    for i in range(pred.shape[0]):
        conf_matrix[pred[i], label[i]] += 1
    df_cm = pd.DataFrame(conf_matrix.numpy(),
                         index=["pred %d" % i for i in range(mat_range)],
                         columns=["label %d" % i for i in range(mat_range)])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="BuPu")
    plt.show()


def get_dataLoader(X, y):
    return DataLoader(dataset=TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=20, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1x48x48
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 16x24x24
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 96, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 32x12x12
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 288, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 64x6x6->128x3x3
        self.conv4 = nn.Sequential(
            nn.Conv2d(288, 500, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 128x3x3->256x2x2
        self.conv5 = nn.Sequential(
            nn.Conv2d(500, 1000, padding=1, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 256x2x2->512x1x1
        self.conv6 = nn.Sequential(
            nn.Conv2d(1000, 2000, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.mlp = nn.Linear(2000, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.mlp(x.reshape(x.shape[0], -1))
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0")
    cnn = CNN().to(device)
    train_data = get_data("./train.csv")
    test_data = get_data("./testMy.csv")
    data_loader = get_dataLoader(train_data[0], train_data[1])

    optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-4)
    for epoch in range(8):
        for i, (imgs, labels) in enumerate(data_loader):
            prediction = cnn(imgs.to(device))
            loss = nn.CrossEntropyLoss()(prediction, labels.to(device))
            prediction_in_test = cnn(torch.from_numpy(np.float32(test_data[0])).to(device)).cpu()
            pred_y = torch.max(prediction_in_test, 1)[1].data.numpy()
            accuracy = float((pred_y == test_data[1]).astype(int).sum()) / float(test_data[1].shape[0])
            if i % 20 == 0:
                # plot_heatmap(pred_y, test_data[1])
                print('i = %d, epoch = %d, loss = %.3f  acc = %.2f' % (i, epoch, loss.cpu().data.numpy(), accuracy))
                state = {'net': cnn.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'i': i}
                torch.save(state, "./model/saver.pkl")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
