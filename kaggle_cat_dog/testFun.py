from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue,
    RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
import cv2

# img = cv2.imread("train/cat.0.jpg")
# cv2.imshow("imgpre", img)
# img = CenterCrop(100, 100, p=1.)(image=img)['image']
# cv2.imshow("imgaft", img)
# cv2.waitKey(0)
import torch
print(torch.__version__)
