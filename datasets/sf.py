import json
import os
from tkinter import Image
from PIL import Image
import numpy as np
from keras.utils import to_categorical
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch
from pathlib import Path
from .coco import make_coco_transforms

class SmartFarm(Dataset):
    def __init__(self, path, image_set='train', transform=None, n_cls=1):
        super().__init__()
        assert image_set in ['train', 'val', 'test'], 'set must be one of train, valid, test'
        if image_set == 'val':
            self.path = os.path.join(path, 'valid')
        else:
            self.path = os.path.join(path, image_set)
        self.json_path = os.path.join(self.path, '_annotations.coco.json')

        self.transform = transform
        self.n_cls = n_cls + 1              # add background class
        self.ids = list(self.image2id().keys())

    def __len__(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            image_mapping = self.image2id()
        return len(image_mapping)

    def __getitem__(self, index):
        idx = self.ids[index]
        img, target = self.read_json(self.json_path, idx)
        # img = transforms.ToTensor()(img)
        if self.transform is not None:
            img, target = self.transform(img, target)
        if target['boxes'].shape[0] == 0:
            print('false')
        else:
            print('true')
        return img, target

    def label2id(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            mapping = {}
            for item in data:
                mapping[item['id']] = item['name']
        return mapping

    def image2id(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            mapping = {}
            for item in data['images']:
                mapping[item['id']] = item['file_name']
        return mapping

    def read_json(self, path, idx):
        with open(path, 'r') as f:
            data = json.load(f)
            image_mapping = self.image2id()
            target = {}
            bboxes, classes, image_files, area, iscrowd = [], [], [], [], []
            for ann in data['annotations']:
                if ann['image_id'] == idx:
                    image_files.append(image_mapping[ann['image_id']])
                    bboxes.append(ann['bbox'])
                    classes.append(ann['category_id'])
                    area.append(ann['area'])
                    iscrowd.append(ann['iscrowd'])
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            tmp = bboxes.reshape(-1, 2, 2)
            left_corner = tmp[:, 0]
            right_corner = tmp[:, 0] + tmp[:, 1]
            bboxes = torch.cat((left_corner, right_corner), dim=-1)
            classes = torch.as_tensor(classes, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd)
            # print(len(bboxes), len(classes), len(iscrowd))
            target["boxes"] = bboxes
            target["labels"] = classes
            target["image_id"] = torch.tensor(idx)
            target["area"] = area
            target["iscrowd"] = iscrowd
            image = Image.open(os.path.join(self.path, image_files[0]))
            w, h = image.size
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["size"] = torch.as_tensor([int(h), int(w)])

            return image, target

def build(image_set, args):
    path = Path(args.other_dataset_path)
    assert path.exists(), f'provided SF path {path} does not exist'
    dataset = SmartFarm(path, image_set, transform=make_coco_transforms(image_set))
    return dataset

if __name__ == '__main__':
    MyDataset = SmartFarm('/Users/shijunshen/Documents/Code/dataset/Smart_Farm_Detection.v1i.coco')
    print(MyDataset[0])