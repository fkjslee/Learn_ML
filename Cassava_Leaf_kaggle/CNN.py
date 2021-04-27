import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def get_data(path):
    pd_file = pd.read_csv(os.path.join(path, os.listdir(path)[0]))
    labels = pd_file.values[:, 1]
    img_names = pd_file.values[:, 0]
    X = []
    # idx = 0
    for img_name in img_names:
        img = cv2.imread(os.path.join("./data/train_images", img_name), 0)
        # cv2.imshow("img", img)
        # print(labels[idx])
        # idx += 1
        # cv2.waitKey(0)
        # continue
        img_size = (48, 48)
        img = cv2.warpAffine(img, np.float32([[img_size[1] / img.shape[1], 0, 0], [0, img_size[0] / img.shape[0], 0]]),
                             img_size)
        img = img.reshape((1, img.shape[0], img.shape[1]))
        X.append(img / 255.0)
        # if len(X) > 100:
        #     break
    return np.float32(X), np.int64(labels[0: len(X)])


def get_dataLoader(X, y):
    return DataLoader(dataset=TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=20, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1x48x48
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 32x24x24
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 96, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 96x12x12
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 288, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 288x6x6->500x3x3
        self.conv4 = nn.Sequential(
            nn.Conv2d(288, 500, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # # 128x3x3->256x2x2
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(500, 1000, padding=1, kernel_size=2, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        # # 256x2x2->512x1x1
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(1000, 2000, padding=1, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        self.mlp = nn.Linear(4500, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = self.mlp(x.reshape(x.shape[0], -1))
        return x


def plot_heatmap(pred, label):
    import seaborn as sn
    import matplotlib.pyplot as plt
    import pandas as pd
    mat_range = torch.max(torch.max(torch.from_numpy(pred)), torch.max(torch.from_numpy(label))).data.numpy() + 1
    conf_matrix = torch.zeros(mat_range, mat_range, dtype=torch.int64)
    for i in range(pred.shape[0]):
        conf_matrix[pred[i], label[i]] += 1
    df_cm = pd.DataFrame(conf_matrix.numpy(),
                         index=["pred %d" % i for i in range(5)],
                         columns=["label %d" % i for i in range(5)])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="BuPu")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0")
    cnn = CNN().to(device)
    train_data = get_data("./data/train")
    test_data = get_data("./data/valid")
    data_loader = get_dataLoader(train_data[0], train_data[1])

    optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-4)
    print('start to train')
    for epoch in range(8):
        for i, (imgs, labels) in enumerate(data_loader):
            prediction = cnn(imgs.to(device))
            loss = nn.CrossEntropyLoss()(prediction, labels.to(device))
            prediction_in_test = cnn(torch.from_numpy(np.float32(test_data[0])).to(device)).cpu()
            pred_y = torch.max(prediction_in_test, 1)[1].data.numpy()
            accuracy = float((pred_y == test_data[1]).astype(int).sum()) / float(test_data[1].shape[0])
            if i % 20 == 0:
                print('i = %d, epoch = %d, loss = %.3f  acc = %.2f' % (i, epoch, loss.cpu().data.numpy(), accuracy))
                state = {'net': cnn.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'i': i}
                torch.save(state, "./model/saver.pkl")
            if i % 200 == 0:
                plot_heatmap(pred_y, test_data[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
