import numpy as np
import cv2
import pandas as pd
import os
import re

root_path = "G:\GWHD"

if __name__ == "__main__":
    train = pd.read_csv(os.path.join(root_path, "train.csv"))
    train = train[:10]
    for image_id, width, height, bbox, source in train.values:
        img = cv2.imread(os.path.join(root_path, 'train', image_id + ".jpg"))
        bbox = re.sub('[\[|\]]', ' ', bbox)
        bbox = np.fromstring(bbox, sep=', ', dtype=np.float32).astype(np.int32)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255))
        cv2.imshow("pic", img)
        cv2.waitKey(0)
