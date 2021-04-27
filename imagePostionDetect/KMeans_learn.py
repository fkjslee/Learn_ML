import matplotlib.pyplot as plt
import cv2
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import os
import numpy as np
import shutil


def walk_files(path='.', endpoint=''):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(endpoint):
                file_list.append(file_path)
    return file_list


def remove_files(path):
    train_list = walk_files(path, "")
    for file_name in train_list:
        os.remove(file_name)


# X为样本特征，Y为样本簇类别，共1000个样本，每个样本2个特征，对应x和y轴，共4个簇，
# 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
# X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
#                   cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9)
#
# print(X.shape)
#
# plt.scatter(X[:, 0], X[:, 1], marker='o')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
# plt.show()
#
#
# y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()

if __name__ == "__main__":
    k_means_img_dir = "./k_means_img"
    if not os.path.exists(k_means_img_dir):
        os.makedirs(k_means_img_dir)
    remove_files(k_means_img_dir)

    img_list = walk_files("C:\\Users\\fkjslee\\Desktop\\0\\0\\0", "")
    X = np.float32([]).reshape(0, 100 * 100 * 3)
    for file_name in img_list:
        img = cv2.imread(file_name)
        img = cv2.warpAffine(img, np.float32([[100.0 / img.shape[1], 0, 0], [0, 100.0 / img.shape[0], 0]]), (100, 100))
        X = np.row_stack((X, img.reshape(-1)))

    print('read over')
    y_pred = KMeans(n_clusters=5).fit_predict(X)
    idx = 0
    for idx in range(len(y_pred)):
        print('idx', idx)
        dir_name = os.path.join(k_means_img_dir, "_" + str(y_pred[idx]))
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        file_name = os.path.join(dir_name, str(idx) + ".png")
        img = X[idx].reshape(100, 100, 3).astype(np.uint8)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        shutil.copy(img_list[idx], file_name)
