import torch
import os
import numpy as np
import cv2


class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


def get_batch():
    train_path = "./testMy"
    file_names = os.listdir(train_path)
    batchX = []
    batchY = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(train_path, file_name), 0)
        img = cv2.warpAffine(img, np.float32([[28 / img.shape[1], 0, 0], [0, 28 / img.shape[0], 0]]), (28, 28))
        img = img.astype(np.float32)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                img[r][c] = img[r][c] / 256.0
        if file_name.startswith("cat"):
            batchX.append(img)
            batchY.append(1)
        else:
            batchX.append(img)
            batchY.append(0)
    batchX = torch.from_numpy(np.float32(batchX).reshape(-1, 1, 28, 28))
    batchY = torch.from_numpy(np.int64(batchY))
    return batchX, batchY


model = CNNnet()
model.load_state_dict(torch.load('model/saver.pth.tar'))

batch_x, batch_y = get_batch()
out = model(batch_x)
# print('test_out:\t',torch.max(out,1)[1])
# print('test_y:\t',test_y)
accuracy = torch.max(out, 1)[1].numpy() == batch_y.numpy()
print('accuracy:\t', accuracy.mean())
print(torch.cuda.is_available())
