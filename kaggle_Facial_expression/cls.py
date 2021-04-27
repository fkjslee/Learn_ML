import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import pandas as pd
import CNN
from torch.utils.data import DataLoader, TensorDataset


def remove_files(path: str):
    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))


# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
def get_data(path):
    train_pd = pd.read_csv(path)
    labels = train_pd.values[:, 0]
    imgs = train_pd.values[:, 1]
    X = []
    for i in range(imgs.shape[0]):
        img = np.fromstring(imgs[i], dtype=np.float32, sep=' ').reshape((1, 48, 48))
        img = img / 255.0
        X.append(img)
    return np.float32(X), np.int64(labels)


device = torch.device("cuda:0")
cnn = CNN.CNN().to(device)

ck = torch.load("./model/saver.pkl")
cnn.load_state_dict(ck['net'])
test_data = get_data("./testMy.csv")

prediction_in_test = cnn(torch.from_numpy(np.float32(test_data[0])).to(device)).cpu()
pred_y = torch.max(prediction_in_test, 1)[1].data.numpy()
accuracy = float((pred_y == test_data[1]).astype(int).sum()) / float(test_data[1].shape[0])
print("acc = %.3f" % accuracy)


facial = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
remove_files("./right_cls")
remove_files("./wrong_cls")
for j in range(pred_y.shape[0]):
    img = test_data[0][j][0] * 255
    real_label = facial[test_data[1][j]]
    pred_label = facial[pred_y[j]]
    if pred_label != real_label:
        cv2.imwrite("wrong_cls//%d_pred=%s_real=%s.png" % (j, pred_label, real_label), img)
    else:
        cv2.imwrite("right_cls//%d_pred=%s_real=%s.png" % (j, pred_label, real_label), img)
