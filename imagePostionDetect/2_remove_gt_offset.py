import os
import numpy as np


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    dir_path = "./1"
    for dir_name in os.listdir(dir_path):
        relative_dir_name = os.path.join(dir_path, dir_name, "std")
        gt_file_name = os.path.join(relative_dir_name, "groundTruth.txt")
        print(gt_file_name)
        content = []
        with open(gt_file_name) as f_gt:
            lines = f_gt.readlines()
            for idx in range(len(lines) // 2):
                line = lines[idx * 2 + 1].replace("\n", "")
                content.append(np.fromstring(line, sep=' ', dtype=np.float32).reshape(3, 3))
            offset = np.linalg.inv(content[0])
            for idx in range(len(content)):
                content[idx] = np.matmul(offset, content[idx])
        with open(gt_file_name[:gt_file_name.rindex(".txt")] + "_off.txt", 'w') as f_gt:
            for idx in range(len(content)):
                f_gt.write(str(idx+1) + "\n")
                f_gt.write(str(content[idx].reshape(1, -1).tolist()).replace("[", "").replace("]", "").replace(",", "") + "\n")
