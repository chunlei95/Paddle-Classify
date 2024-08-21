import argparse
import os.path
from glob import glob
from shutil import copy

import cv2
import numpy as np
import tifffile

from utils.labelme_utils import main


def split_image(img_path, save_path):
    img_arr = tifffile.imread(img_path)
    img_height, img_width = img_arr.shape[0:-1]
    assert img_arr.shape[-1] == 4
    if img_arr.shape[-1] == 4:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
    file_path = os.path.split(img_path)[-1]
    filename, ext = os.path.splitext(file_path)
    save_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cols, rows = img_width // 512, img_height // 512
    for i in range(rows):
        for j in range(cols):
            sub_num = i * (cols) + j + 1
            img_save_path = os.path.join(str(save_path), filename + '_' + str(sub_num) + ext)
            sub_arr = img_arr[i * 512: (i + 1) * 512, j * 512: (j + 1) * 512, :]
            cv2.imwrite(str(img_save_path), sub_arr)


def split_dataset(root_path, test_ratio, labeled_ratio=1.0 / 16):
    imgs = glob(root_path + '/*')
    counts = len(imgs)
    test_size = int(counts * test_ratio)
    # np.random.seed(42)
    np.random.shuffle(imgs)
    test_paths = imgs[:test_size]
    train_paths = imgs[test_size:]
    labeled_train_size = int(len(train_paths) * labeled_ratio)
    labeled_train_paths = train_paths[:labeled_train_size]
    unlabeled_train_paths = train_paths[labeled_train_size:]
    copy_file(test_paths, 'D:/datasets/Cropland_Identity/Cropland_Identity/img_dir/val')
    copy_file(labeled_train_paths, 'D:/datasets/Cropland_Identity/Cropland_Identity/img_dir/train_labeled')
    copy_file(unlabeled_train_paths, 'D:/datasets/Cropland_Identity/Cropland_Identity/img_dir/train_unlabeled')


def copy_file(file_paths, dest_folder):
    if os.path.isdir(file_paths):
        for i in range(len(file_paths)):
            copy(file_paths[i], dest_folder)
    else:
        copy(file_paths, dest_folder)


def extract_label(label_root, save_path):
    label_folders = glob(label_root + '/*')
    label_paths = [folder + '/label.png' for folder in label_folders if os.path.isdir(folder)]
    for p in label_paths:
        label_name = p.split('/')[-2]
        label_name = label_name.split('\\')[-1]
        save_name = label_name + '.png'
        label_save_path = os.path.join(save_path, save_name)
        copy_file(p, label_save_path)


def label_map(label_root):
    label_paths = glob(label_root + '/*')
    for p in label_paths:
        label_arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        label_value = label_arr.max()
        label_ = label_arr.copy()
        label_arr[label_ == label_value] = 1
        cv2.imwrite(p, label_arr)


if __name__ == '__main__':
    # img_path = r'D:/datasets/Cropland_Identity/cropland_identity_datasource/Cropland_Identity/region6'
    # split_image(img_path, 'D:/datasets/cropland_identity/Cropland_Identity')
    # split_dataset(img_path, 0.2, 1.0 / 16)
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()
    file_path = args.json_file
    json_paths = glob(file_path + '/*')
    for path in json_paths:
        main(path, args)
    json_label_path = 'D:/datasets/Cropland_Identity/label_json/train_labeled'
    save_label_path = 'D:/datasets/Cropland_Identity/Cropland_Identity/ann_dir/train_labeled'
    extract_label(json_label_path, save_label_path)
    label_map(save_label_path)
    # img_path = 'D:/datasets/Cropland_Identity/Cropland_Identity/ann_dir/train_labeled/region1_67.png'
    # img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # print(img_arr.max())
