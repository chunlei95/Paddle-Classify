import math
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from paddle.io import Dataset
from sklearn.cluster import KMeans

from datasets.augment import create_train_dataset


class InsectPestDataset(Dataset):

    def __init__(self, data_root, augment_root, val_ratio=0.2, mode='train', transforms=None):
        super().__init__()
        self.data_root = data_root
        self.CLASS_NUM = 0
        self.CLASS_NAME = []
        self.image_list = []
        self.label_list = []
        self.transforms = transforms
        img_list = []
        ann_list = []
        img_dict = {}
        count_dict = {}
        self._read_data(data_root)
        print(self.CLASS_NUM)
        for i in range(self.CLASS_NUM):
            class_name = self.CLASS_NAME[i]
            class_images = [p for p in self.image_list if os.path.split(p.split(self.data_root)[-1])[0] == class_name]
            class_images.sort()
            img_num = len(class_images)
            np.random.seed(42)
            np.random.shuffle(class_images)
            val_num = math.ceil(img_num * val_ratio)
            if mode == 'val':
                img_list.extend(class_images[:val_num])
                # for p in paths[:val_num]:
                #     img_dict.update({p: paths})
                ann_list.extend([i] * val_num)
                # count_dict.update({class_name: val_num})
            elif mode == 'test':
                img_list.extend(class_images)
                ann_list.extend([i] * len(class_images))
            else:
                img_list.extend(class_images[val_num:])
                for p in class_images[val_num:]:
                    img_dict.update({p: class_name})
                ann_list.extend([i] * (img_num - val_num))
                count_dict.update({class_name: (img_num - val_num)})
        self.image_list = img_list
        self.label_list = ann_list
        del img_list, ann_list
        print(count_dict)
        if mode == 'train':
            if not os.path.exists(augment_root):
                os.makedirs(augment_root)
            data_paths = glob(augment_root + '/*')
            if data_paths is None or len(data_paths) == 0:
                create_train_dataset(augment_root, 500, count_dict, img_dict)
            self.image_list = []
            self.label_list = []
            self.CLASS_NUM = 0
            self.CLASS_NAME = []
            img_list = []
            ann_list = []
            count_dict = {}
            self._read_data(augment_root)
            print(self.CLASS_NUM)
            for i in range(self.CLASS_NUM):
                class_name = self.CLASS_NAME[i]
                class_images = [p for p in self.image_list if
                                os.path.split(p.split(self.data_root)[-1])[0] == class_name]
                img_list.extend(class_images)
                ann_list.extend([i] * len(class_images))
                count_dict.update({class_name: len(class_images)})
            self.image_list = img_list
            self.label_list = ann_list
            del img_list, ann_list
            print(count_dict)
        del count_dict, img_dict
        # print(len(self.image_list))

    def _read_data(self, data_path):
        if os.path.isdir(data_path):
            paths = glob(data_path + '/*')
            for p in paths:
                self._read_data(p)
        else:
            class_name = os.path.split(data_path.split(self.data_root)[-1])[0]
            if class_name not in self.CLASS_NAME:
                self.CLASS_NAME.append(class_name)
                self.CLASS_NUM += 1
            self.image_list.append(data_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        ann = self.label_list[item]
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, ann


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


if __name__ == '__main__':
    data_root = '/media/humrobot/Data/datasets/农作物病虫害数据集'
    data = pd.DataFrame(columns=['img_path', 'class_name', 'class_id', 'mode'])
    read_data(data_root)
    train_class_nums = []
    idx = 0
    for cn in class_names:
        class_images = [p for p in img_list if os.path.split(p.split(data_root)[-1])[0] == cn]
        class_images.sort()
        val_num = int(len(class_images) * 0.2)
        np.random.seed(42)
        np.random.shuffle(class_images)
        val_imgs = class_images[:val_num]
        train_imgs = class_images[val_num:]
        train_class_num = len(train_imgs)
        train_class_nums.append(train_class_num)
        for p in val_imgs:
            data.loc[len(data.index)] = [p, cn, idx, 'val']  # 路径 类别名称 类别id 训练集还是测试集
        for p in train_imgs:
            data.loc[len(data.index)] = [p, cn, idx, 'train']
    data.to_csv('pest_and_disease.csv', index=False)
    kmeans = KMeans(n_clusters=5)

    class_nums_array = np.array(train_class_nums).reshape(-1, 1)
    predicts = kmeans.fit_predict(class_nums_array)
    predicts = predicts.tolist()
    print(kmeans.cluster_centers_)
    collect_data = pd.DataFrame(dict(class_name=class_names, class_num=train_class_nums, cluster_index=predicts))
    collect_data.to_csv('pest_and_disease_analyse.csv', index=False)
