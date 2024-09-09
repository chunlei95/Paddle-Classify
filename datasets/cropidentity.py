import math
from glob import glob

import numpy as np
from PIL import Image
from paddle.io import Dataset

from datasets.augment import create_train_dataset

class_name2id = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'q': 16,
    'r': 17,
    's': 18
}


class CropIdentityDataset(Dataset):

    def __init__(self, data_root, augment_root, val_ratio=0.2, mode='train', transforms=None):
        super().__init__()
        self.image_list = []
        self.label_list = []
        self.transforms = transforms
        img_dict = {}
        count_dict = {}
        data_paths = glob(data_root + '/*')
        for path in data_paths:
            class_name = path.split(data_root)[-1]
            class_name = class_name.split('\\')[-1]
            class_id = class_name2id[class_name]
            image_paths = glob(path + '/*')
            img_num = len(image_paths)
            np.random.seed(1234)
            np.random.shuffle(image_paths)
            val_num = math.ceil(img_num * val_ratio)
            if mode == 'val':
                self.image_list.extend(image_paths[:val_num])
                for p in image_paths[:val_num]:
                    img_dict.update({p: class_name})
                self.label_list.extend([class_id] * val_num)
                count_dict.update({class_name: val_num})
            elif mode == 'test':
                self.image_list.extend(image_paths)
                self.label_list.extend([class_id] * len(image_paths))
            else:
                self.image_list.extend(image_paths[val_num:])
                for p in image_paths[val_num:]:
                    img_dict.update({p: class_name})
                self.label_list.extend([class_id] * (img_num - val_num))
                count_dict.update({class_name: (img_num - val_num)})
        print(count_dict)
        if mode == 'train':
            data_paths = glob(augment_root + '/*')
            if data_paths is None or len(data_paths) == 0:
                create_train_dataset(augment_root, 1000, count_dict, img_dict)
            self.image_list = []
            self.label_list = []
            count_dict = {}
            data_paths = glob(augment_root + '/*')
            for p in data_paths:
                class_name = p.split(augment_root)[-1]
                class_name = class_name.split('\\')[-1]
                class_id = class_name2id[class_name]
                image_paths = glob(p + '/*')
                self.image_list.extend(image_paths)
                self.label_list.extend([class_id] * len(image_paths))
                count_dict.update({class_name: len(image_paths)})
            print(count_dict)
        del count_dict, img_dict
        print(len(self.image_list))

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
