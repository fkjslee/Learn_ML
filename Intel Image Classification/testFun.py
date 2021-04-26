import threading
from matplotlib import pyplot as plt
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import torch.utils.data as Data
import torchvision
import timm


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue,
    RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


class IntelDataset(Dataset):
    def __init__(self, df, data_root):
        super().__init__()
        self.df = df
        self.data_root = data_root

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = os.path.join(self.df.loc[index]['image_id'])
        img = cv2.imread(path)
        if img.shape != (150, 150, 3):
            img = cv2.warpAffine(img, np.float32([[150. / img.shape[0], 0, 0], [0, 150. / img.shape[1], 0]]), (150, 150))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = get_train_transforms()(image=img)['image']

        return img, self.df.label[index]


class EfficientNet(nn.Module):
    def __init__(self, model_arch, n_class):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


def prepare_dataloader(df, trn_idx, val_idx, data_root):
    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    print(valid_.head(20))
    train_ds = IntelDataset(train_, data_root)
    valid_ds = IntelDataset(valid_, data_root)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
        # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader


def train_one_epoch(train_loader, device, model, optimizer):
    model.train()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        prediction = model(imgs)
        loss = nn.CrossEntropyLoss().to(device)(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss = ', loss)


def valid_one_epoch(valid_loader, device, model):
    model.eval()
    preds = []
    reals = []
    for i, (imgs, labels) in enumerate(valid_loader):
        prediction_in_test = model(imgs.to(device))
        pred_y = torch.max(prediction_in_test.cpu(), 1)[1].data.numpy()
        preds.extend(pred_y)
        reals.extend(labels.data.numpy())
    print(preds)
    print(reals)
    preds = np.int32(preds)
    reals = np.int32(reals)
    print('acc = ', float((preds == reals).astype(int).sum()) / float(reals.shape[0]))


CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 150,
    'epochs': 10,
    'train_bs': 16,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0'
}

train_root_path = "./data/seg_train/seg_train"
if __name__ == "__main__":
    seed_all(CFG['seed'])

    labelMap = {}
    for (labelID, entity) in enumerate(os.listdir(train_root_path)):
        labelMap[entity] = labelID

    paths = []
    labels = []
    for filepath, dirnames, filenames in os.walk(train_root_path):
        for filename in filenames:
            absname = os.path.join(filepath, filename)
            paths.append(absname)
            for entity in labelMap.keys():
                if absname.find(entity) != -1:
                    labels.append(labelMap[entity])

    train = pd.DataFrame(data={"image_id": paths, "label": labels})
    # train = train[0:20]
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(
        np.arange(train.shape[0]), train.label.values)

    for fold, (train_idx, valid_idx) in enumerate(folds):
        if fold > 0:
            break

        print('Training with {} started'.format(fold))

        train_loader, val_loader = prepare_dataloader(train, train_idx, valid_idx, data_root=train_root_path)
        device = torch.device(CFG['device'])

        model = EfficientNet('tf_efficientnet_b4_ns', 6).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        print('start to train')

        for epoch in range(CFG['epochs']):
            print('epoch = %d' % epoch)
            train_one_epoch(train_loader, device, model, optimizer)
            valid_one_epoch(val_loader, device, model)
