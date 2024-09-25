import math
import os
import uuid
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

from datasets.augment import data_augment

img_list = []
class_names = []
data_root = ''


def read_data(data_path):
    if os.path.isdir(data_path):
        paths = glob(data_path + '/*')
        for p in paths:
            read_data(p)
    else:
        class_name = os.path.split(data_path.split(data_root)[-1])[0]
        if class_name not in class_names:
            class_names.append(class_name)
        img_list.append(data_path)


def split_data(path):
    pass


def augment_train_dataset(ori_images, class_name, save_root, aug_num):
    if len(ori_images) >= aug_num:
        return ori_images[:aug_num]
    results = []
    results.extend(ori_images)
    diff = aug_num - len(ori_images)
    aug_loop = tqdm(range(diff), total=diff, colour='green', leave=True, unit='img')
    for i in aug_loop:
        rand_n = np.random.randint(0, len(ori_images))
        p = ori_images[rand_n]
        _, file_name = os.path.split(p)
        new_path = save_root + class_name
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_path = new_path + '/' + file_name

        img = Image.open(p)
        img_arr = np.array(img)
        if img_arr.shape[-1] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
        else:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        img_h, img_w, _ = img_arr.shape
        new_img = data_augment(img_arr.copy(), img_w, img_h)
        name, ext = os.path.splitext(new_path)
        name = name + '_' + uuid.uuid4().hex
        save_img_path = name + ext
        cv2.imencode('.jpg', new_img)[1].tofile(save_img_path)
        results.append(save_img_path)
        # cv2.imwrite(save_img_path, new_img)
        aug_loop.set_description('Augment images of class {}'.format(class_name))
    return results


def prepare_dataset(ori_root, save_root, aug_strategy='center'):
    # if os.path.exists(save_root):
    #     os.removedirs(save_root)
    read_data(ori_root)
    data = pd.DataFrame(
        columns=['img_path', 'class_name', 'class_id', 'mode', 'cluster_id', 'cluster_center', 'cluster_class_id'])
    # data = pd.DataFrame(columns=['img_path', 'class_name', 'class_id', 'mode'])
    cid = 0

    # ****************************************************************************************
    train_nums = []
    for cn in class_names:
        class_images = [p for p in img_list if os.path.split(p.split(data_root)[-1])[0] == cn]
        val_num = math.ceil(len(class_images) * 0.2)
        train_num = len(class_images) - val_num
        train_nums.append(train_num)
    train_nums_arr = np.array(train_nums).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=2546)
    cluster_ids = kmeans.fit_predict(train_nums_arr)
    cluster_id_list = cluster_ids.tolist()
    cluster_centers = kmeans.cluster_centers_
    cluster_center_list = cluster_centers.tolist()
    # *****************************************************************************************
    flag = {i: 0 for i in range(len(cluster_centers))}
    for i, cn in enumerate(class_names):
        class_images = [p for p in img_list if os.path.split(p.split(data_root)[-1])[0] == cn]
        class_images.sort()
        np.random.seed(42)
        np.random.shuffle(class_images)
        val_num = math.ceil(len(class_images) * 0.2)

        val_imgs = class_images[:val_num]
        for p in val_imgs:
            # -1表示不需要聚类中心参与
            # data.loc[len(data.index)] = [p, cn, cid, 'val']
            data.loc[len(data.index)] = [p, cn, cid, 'val', cluster_id_list[i],
                                         cluster_center_list[cluster_id_list[i]][0], flag[cluster_id_list[i]]]
        train_imgs = class_images[val_num:]

        if aug_strategy == 'max':
            cluster_ids_sel = cluster_ids == cluster_id_list[i]
            train_nums_arr_sel = train_nums_arr[cluster_ids_sel]
            aug_num = max(train_nums_arr_sel).item()
        else:
            aug_num = int(cluster_center_list[cluster_id_list[i]][0])
        train_imgs = augment_train_dataset(train_imgs, cn, save_root, aug_num)
        # train_imgs = augment_train_dataset(train_imgs, cn, save_root, 200)
        for p in train_imgs:
            data.loc[len(data.index)] = [p, cn, cid, 'train', cluster_id_list[i],
                                         cluster_center_list[cluster_id_list[i]][0], flag[cluster_id_list[i]]]
        flag[cluster_id_list[i]] += 1
        cid += 1

    data.to_csv('./datasets/insect_pest.csv', index=False)


if __name__ == '__main__':
    data_root = '/media/humrobot/Data/datasets/pest_and_disease_new/insect_pest'
    prepare_dataset(data_root, '/home/humrobot/datasets/aug_insect_pest_train')
