import os
import cv2
import numpy as np


def show_pics(pics: list):
    side = np.ceil(np.sqrt(len(pics)))
    if len(pics) == 0:
        return
    img = np.uint8([0] * int(pics[0].shape[0] * pics[0].shape[1] * (side * side)) * 3).reshape((int(pics[0].shape[1] * side), -1, 3))
    for i in range(len(pics)):
        x = int((i % side) * pics[0].shape[1])
        y = int((i // side) * pics[0].shape[0])
        img[y:y+pics[0].shape[1], x:x+pics[0].shape[0]] = pics[i]
    cv2.imshow("img", img)
    cv2.waitKey(0)


imgs = []
path = "./wrong_cls"
for file_name in os.listdir(path):
    imgs.append(cv2.imread(os.path.join(path, file_name)))
    if len(imgs) > 60:
        break
show_pics(imgs)
