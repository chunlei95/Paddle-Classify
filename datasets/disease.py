import pandas as pd
from PIL import Image
from paddle.io import Dataset


class DiseaseDataset(Dataset):

    def __init__(self, file_path, mode='train', cluster_id=0, transforms=None):
        super().__init__()
        self.image_list = []
        self.label_list = []
        self.transforms = transforms
        data = pd.read_csv(file_path)
        data = data[data['mode'] == mode]
        data = data[data['cluster_id'] == cluster_id]
        self.class_num = len(data.groupby('class_id'))
        self.image_list = data['img_path'].values.tolist()
        self.label_list = data['cluster_class_id'].values.tolist()

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
