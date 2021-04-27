import os
import numpy as np
import shutil
import random


def walk_files(path='.', endpoint='.png'):
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


def distribute_files(src, dst):
    for dir_name in src:
        relative_gt_name = os.path.join(dir_name, "groundTruth_off.txt")
        with open(os.path.join(dir_name, "fragmentList.txt"), 'r') as f_frag_list:
            if len(f_frag_list.readlines()) < 20:
                continue
        with open(relative_gt_name, 'r') as f_gt:
            lines = f_gt.readlines()
            trans = []
            for i in range(20):
                trans.append(np.fromstring(lines[i*2+1], np.float32, sep=' '))
        max_x = np.max([t[2] for t in trans])
        max_y = np.max([t[5] for t in trans])
        for i in range(20):
            old_fragment_name = os.path.join(dir_name, "fragment_%04d.png" % (i+1))
            if trans[i][2] < max_x / 3:
                pos_x = 0
            elif trans[i][2] < max_x * 2 / 3:
                pos_x = 1
            else:
                pos_x = 2
            if trans[i][5] < max_y / 3:
                pos_y = 0
            elif trans[i][5] < max_y * 2 / 3:
                pos_y = 1
            else:
                pos_y = 2
            new_fragment_name = os.path.join(dst, dir_name[4:-4] + "_%02d_%d_%d.png" % (i, pos_x, pos_y))
            # print(old_fragment_name, new_fragment_name)
            shutil.copy(old_fragment_name, new_fragment_name)


if __name__ == "__main__":
    remove_files("./test")
    remove_files("./train")

    dirpath = "./1"
    dir_list = os.listdir(dirpath)
    dir_list = [os.path.join(dirpath, s, "std") for s in dir_list]
    random.shuffle(dir_list)
    distribute_files(dir_list[:int(0.7 * len(dir_list))], "./train")
    distribute_files(dir_list[int(0.7 * len(dir_list)):], "./test")


