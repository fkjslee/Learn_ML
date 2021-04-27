import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import pandas as pd
import CNN
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageChops, ImageFilter
from matplotlib import pyplot as plt
from torch.autograd import Variable


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
modulelist = list()
modulelist.append(cnn.conv1[0])
modulelist.append(cnn.conv1[1])
modulelist.append(cnn.conv1[2])
modulelist.append(cnn.conv2[0])
modulelist.append(cnn.conv2[1])
modulelist.append(cnn.conv2[2])
modulelist.append(cnn.conv3[0])
modulelist.append(cnn.conv3[1])
modulelist.append(cnn.conv3[2])
modulelist.append(cnn.conv4[0])
modulelist.append(cnn.conv4[1])
modulelist.append(cnn.conv4[2])
modulelist.append(cnn.conv5[0])
modulelist.append(cnn.conv5[1])
modulelist.append(cnn.conv5[2])
modulelist.append(cnn.conv6[0])
modulelist.append(cnn.conv6[1])
modulelist.append(cnn.conv6[2])
modulelist.append(cnn.mlp)
# print(modulelist)
#
# prediction_in_test = cnn(torch.from_numpy(np.float32(test_data[0])).to(device)).cpu()
# pred_y = torch.max(prediction_in_test, 1)[1].data.numpy()
# accuracy = float((pred_y == test_data[1]).astype(int).sum()) / float(test_data[1].shape[0])
# print("acc = %.3f" % accuracy)

def deprocess(image):
    return image * torch.tensor([0.229, 0.224, 0.225]).cuda() + torch.tensor([0.485, 0.456, 0.406]).cuda()


# 这是deep dream的实际代码，特定层的梯度被设置为等于该层的响应，这导致了该层响应最大化。换句话说，我们正在增强一层检测到的特征，对输入图像（octaves）应用梯度上升算法。
def dd_helper(image, layer, iterations, lr):
    # 一开始的输入是图像经过预处理、在正数第一个维度上增加一个维度以匹配神经网络的输入、传到GPU上
    ori_img = np.array(image).astype(np.float32)
    np_img = cv2.warpAffine(ori_img / 255.0, np.float32([[48 / ori_img.shape[1], 0, 0], [0, 48 / ori_img.shape[0], 0]]), (48, 48), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    np_img = np.swapaxes(np_img, 0, 2)
    np_img = np.swapaxes(np_img, 1, 2)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(np_img.shape[1]):
        for j in range(np_img.shape[2]):
            for k in range(3):
                np_img[k, i, j] = (np_img[k, i, j] - mean[k]) / std[k]
    input = Variable(torch.from_numpy(np_img).unsqueeze(0).cuda(), requires_grad=True)
    # vgg梯度清零
    cnn.zero_grad()
    # 开始迭代
    for i in range(iterations):
        # 一层一层传递输入
        out = input
        for j in range(len(modulelist) - 1):
            out = modulelist[j](out)
        out = out.reshape(out.shape[0], -1)
        out = modulelist[len(modulelist)-1](out)
        # 损失是输出的范数
        loss = out.norm()
        # 损失反向传播
        loss.backward()
        # 输入的数据是上次迭代时的输入数据+学习率×输入的梯度
        input.data = input.data + lr * input.grad.data
    # 将从网络结构中取出的输入数据的第一个维度去掉
    input = input.data.squeeze()
    # 矩阵转置
    input.transpose_(0, 1)
    input.transpose_(1, 2)
    # 将输入逆标准化后强制截断在0到1的范围内
    tm = deprocess(input)
    input = np.clip(tm.cpu(), 0, 1)
    # 得到像素值为0到255的图像
    im = Image.fromarray(np.uint8(input * 255))
    return im


# 这是一个递归函数，用于创建octaves，并且将由一次递归调用生成的图像与由上一级递归调用生成的图像相融合
def deep_dream_vgg(image, layer, iterations, lr, octave_scale, num_octaves):
    if num_octaves > 0:
        image1 = image.filter(ImageFilter.GaussianBlur(2))
        # 判断是否缩放
        if image1.size[0] / octave_scale < 1 or image1.size[1] / octave_scale < 1:
            size = image1.size
        else:
            size = (int(image1.size[0] / octave_scale), int(image1.size[1] / octave_scale))
        image1 = image1.resize(size, Image.ANTIALIAS)
        image1 = deep_dream_vgg(image1, layer, iterations, lr, octave_scale, num_octaves - 1)
        size = (image.size[0], image.size[1])
        image1 = image1.resize(size, Image.ANTIALIAS)
        # 将最初输入的图像与合成的相同尺寸大小的图像融合
        image = ImageChops.blend(image, image1, 0.6)
    # 按照dd_helper中的流程生成图像
    img_result = dd_helper(image, layer, iterations, lr)
    img_result = img_result.resize(image.size)
    plt.imshow(img_result)
    plt.show()
    return img_result


sky = Image.open('sky.jpg')
print(sky.size)
deep_dream_vgg(sky, 28, 5, 0.2, 2, 20)


# facial = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# remove_files("./right_cls")
# remove_files("./wrong_cls")
# for j in range(pred_y.shape[0]):
#     img = test_data[0][j][0] * 255
#     real_label = facial[test_data[1][j]]
#     pred_label = facial[pred_y[j]]
#     if pred_label != real_label:
#         cv2.imwrite("wrong_cls//%d_pred=%s_real=%s.png" % (j, pred_label, real_label), img)
#     else:
#         cv2.imwrite("right_cls//%d_pred=%s_real=%s.png" % (j, pred_label, real_label), img)
