import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import random


def get_data(path):
    pd_file = pd.read_csv(os.path.join(path, os.listdir(path)[0]))
    labels = pd_file.values[:, 1]
    img_names = pd_file.values[:, 0]
    X = []
    for img_name in img_names:
        img = cv2.imread(os.path.join("./data/train_images", img_name))
        X.append(img)
    return np.float32(X), np.int64(labels[0: len(X)])


if __name__ == "__main__":
    x = np.float32([1, 2, 1, 4])
    print("%d" % np.sum(x == 1))
