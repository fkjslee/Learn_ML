import numpy as np
import cv2
import os
import pandas as pd
from albumentations import ShiftScaleRotate, Compose
from albumentations.pytorch import ToTensorV2


def scale(img, width, height):
    img = cv2.warpAffine(img, np.float32([[width/img.shape[0], 0, 0], [0, height/img.shape[1], 0]]), (width, height))
    return img


root_path = "./data"
channel = ['red', 'green', 'blue', 'yellow']
if __name__ == "__main__":
    code = int('eNoLCAgIMAEABJkBdQ', base=16)
    code = int('11', 16)
    print(code)

    train = pd.read_csv(os.path.join(root_path, "train.csv"))
    for filename, label in train.values:
        print(filename)
        img = np.zeros((4, 2048, 2048))
        for c in range(len(channel)):
            img[c] = cv2.imread(os.path.join(root_path, 'train', "%s_%s.png" % (filename, channel[c])), 0)
        showImg = img[0:3]
        showImg = ToTensorV2()(image=showImg)['image']
        showImg = ToTensorV2()(image=showImg.data.numpy().astype(np.uint8))['image']  # CHW->HWC
        showImg = cv2.cvtColor(showImg.data.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("show", scale(showImg, 840, 840))
        cv2.waitKey(0)
