import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import svm


def get_train_feature_and_label(train_data):
    train_feature = []
    train_label = []
    for i in range(int(train_data.shape[0] / 18)):
        each_feature = []
        for j in range(18):
            each_feature.extend(train_data[i * 18 + j][3:3 + 8])
        train_feature.append(each_feature)
        train_label.append(train_data[i * 18 + 9][11])
    train_feature = np.float32(train_feature)
    train_label = np.float32(train_label)
    return train_feature, train_label


def get_test_feature_and_label(test_data):
    test_feature = []
    test_label = []
    for i in range(int(test_data.shape[0] / 18)):
        each_feature = []
        for j in range(18):
            each_feature.extend(test_data[i * 18 + j][2:2 + 8])
        test_feature.append(each_feature)
        test_label.append(test_data[i * 18 + 9][10])
    test_feature = np.float32(test_feature)
    test_label = np.float32(test_label)
    return test_feature, test_label


def normalization(train_feature, test_feature):
    l = len(train_feature)
    all_feature = np.row_stack((train_feature, test_feature))
    for i in range(all_feature.shape[1]):
        mean = np.mean(all_feature[:, i])
        std = np.std(all_feature[:, i])
        all_feature[:, i] = (all_feature[:, i] - mean) / std
    train_feature = all_feature[:l]
    test_feature = all_feature[l:]
    return train_feature, test_feature


def get_split_feature_and_label(train_feature, train_label):
    # split train data to 7:3
    test_feature = train_feature[int(0.7 * train_feature.shape[0]):]
    test_label = train_label[int(0.7 * train_label.shape[0]):]
    train_feature = train_feature[:int(0.7 * train_label.shape[0])]
    train_label = train_label[:int(0.7 * train_label.shape[0])]
    return train_feature, train_label, test_feature, test_label


if __name__ == "__main__":
    train_data = pd.read_csv("./data/train.csv", encoding='utf-8')
    print(train_data[train_data['item'].isin(['RAINFALL'])].info)
    test_data = pd.read_csv("./data/test.csv", header=None)
    train_data = train_data.replace("NR", 0)
    test_data = test_data.replace("NR", 0)
    train_data = train_data.values
    test_data = test_data.values

    # get train data
    train_feature, train_label = get_train_feature_and_label(train_data)

    # get_test data
    test_feature, test_label = get_test_feature_and_label(test_data)

    train_feature, train_label, test_feature, test_label = get_split_feature_and_label(train_feature, train_label)

    train_feature, test_feature = normalization(train_feature, test_feature)

    clf = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=1,
        gamma=0,
        reg_lambda=1,

        max_delta_step=0,
        # 生成树时进行的列采样
        colsample_bytree=1,

        min_child_weight=1,

        # 随机种子
        seed=1000)

    clf.fit(train_feature, train_label)
    pred = clf.predict(test_feature)
    err = 0.0
    for i in range(test_label.shape[0]):
        err += np.square(test_label[i] - pred[i])
    err = err / test_label.shape[0]
    print(np.sqrt(err))

    plt.plot(test_label[:50], '-or')
    plt.plot(pred[:50], '-ob')
    # plt.show()
