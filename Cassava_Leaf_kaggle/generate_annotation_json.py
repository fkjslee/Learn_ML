# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 22:12
# @Author  : MaochengHu
# @Email   : wojiaohumaocheng@gmail.com
# @File    : generate_annotation_json.py
# @Software: PyCharm

import os
import json


def get_annotation_dict(input_folder_path, word2number_dict):
    label_dict = {}
    father_file_list = os.listdir(input_folder_path)
    for father_file in father_file_list:
        full_father_file = os.path.join(input_folder_path, father_file)
        son_file_list = os.listdir(full_father_file)
        for image_name in son_file_list:
            label_dict[os.path.join(full_father_file, image_name)] = word2number_dict[father_file]
    return label_dict


def save_json(label_dict, json_path):
    with open(json_path, 'w') as json_path:
        json.dump(label_dict, json_path)
    print("label json file has been generated successfully!")
