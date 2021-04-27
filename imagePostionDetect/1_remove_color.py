import cv2
import os
import numpy as np

if __name__ == "__main__":
    dirpath = "./1"
    for dir_name in os.listdir(dirpath):
        relative_dir_name = os.path.join(dirpath, dir_name, "std")
        print('relative name', relative_dir_name)
        with open(os.path.join(relative_dir_name, "bg_color.txt")) as f_bg_color:
            bg_color = np.fromstring(f_bg_color.readlines()[0], sep=' ', dtype=np.int32)
            bg_color = [bg_color[2], bg_color[1], bg_color[0]]
        for file_name in os.listdir(relative_dir_name):
            relative_file_name = os.path.join(relative_dir_name, file_name)
            if relative_file_name.find("fragment_00") != -1:
                img = cv2.imread(relative_file_name)
                img[np.where(np.all(img == bg_color, axis=2))] = [0, 0, 0]
                cv2.imwrite(relative_file_name, img)
