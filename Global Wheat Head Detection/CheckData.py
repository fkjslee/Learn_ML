import numpy as np
import cv2
import pandas as pd
import os
import re
from sklearn.model_selection import StratifiedKFold

root_path = "G:\GWHD"

if __name__ == "__main__":
    TRAIN_ROOT_PATH = os.path.join(root_path, "train")
    marking = pd.read_csv(os.path.join(root_path, "train.csv"))

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:, i]
    marking.drop(columns=['bbox'], inplace=True)

    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0

    fold_number = 0
    image_ids = df_folds
    TRAIN_ROOT_PATH = os.path.join(root_path, "train")
    for i, (image_id) in enumerate(image_ids.index):
        if image_id != "25d38ec2c":
            continue
        print(image_id)
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = image.astype(np.float32)
        image /= 255.0
        records = marking[marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = boxes.astype(np.int32)
        for bbox in boxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 1.0))
        cv2.imshow("image", image)
        cv2.waitKey(0)

