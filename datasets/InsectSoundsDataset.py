import os.path

from PIL import Image
from paddle.io import Dataset
from glob import glob


class InsectSoundsDataset(Dataset):

    def __init__(self, data_root, augment_root, val_ratio=0.2, mode='train', transforms=None):
        super().__init__()
        self.image_list = []
        self.label_list = []
        self.transforms = transforms
        self.CLASS_NAME = []
        self.CLASS_ID = []
        data_paths = glob(data_root + '/*')





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
