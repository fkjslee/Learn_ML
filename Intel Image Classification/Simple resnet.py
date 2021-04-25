from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import torch.utils.data as Data
import torchvision


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
            # print(path)
            img = cv2.warpAffine(img, np.float32([[150. / img.shape[0], 0, 0], [0, 150. / img.shape[1], 0]]), (150, 150))
            # plt.imshow(img)
            # plt.show()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        return img, self.df.label[index]


class CNN(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cuda:0')):
        super(CNN, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False
        self.mlp = nn.Linear(2048 * 5 * 5, 6)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.mlp(x.reshape(x.shape[0], -1))
        return x


def prepare_dataloader(df, trn_idx, val_idx, data_root='../input/cassava-leaf-disease-classification/train_images/'):
    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

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
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        prediction = model(imgs)
        loss = nn.CrossEntropyLoss()(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def valid_one_epoch(valid_loader, device, model):
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
    'img_size': 512,
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

    # train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')

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
    # train = train[0:200]
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(
        np.arange(train.shape[0]), train.label.values)

    for fold, (train_idx, valid_idx) in enumerate(folds):
        if fold > 0:
            break

        print('Training with {} started'.format(fold))

        print(len(train_idx), len(valid_idx))
        train_loader, val_loader = prepare_dataloader(train, train_idx, valid_idx, data_root=train_root_path)
        device = torch.device(CFG['device'])

        model = CNN().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        print('start to train')

        for epoch in range(7):
            train_one_epoch(train_loader, device, model, optimizer)
            valid_one_epoch(val_loader, device, model)
