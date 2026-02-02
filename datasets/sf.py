import json
import os

from PIL import Image
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .coco import make_coco_transforms

class SmartFarm(Dataset):
    def __init__(self, path, image_set='train', transform=None, num_classes=2):
        super().__init__()
        assert image_set in ['train', 'val', 'test'], 'set must be one of train, valid, test'
        if image_set == 'val':
            self.path = os.path.join(path, 'valid')
        else:
            self.path = os.path.join(path, image_set)
        self.json_path = os.path.join(self.path, '_annotations.coco.json')

        self.transform = transform
        self.num_classes = num_classes  # number of object classes (not including background)
        
        # Load and cache JSON data once during initialization
        print(f"Loading annotations from {self.json_path}...")
        with open(self.json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image_id to filename mapping
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        
        # Build annotations index: image_id -> list of annotations
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # Get list of image IDs
        self.ids = list(self.image_id_to_filename.keys())
        print(f"Loaded {len(self.ids)} images with annotations cached in memory.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        img, target = self.load_image_and_target(idx)
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def get_num_classes_from_json(self):
        """Get the actual number of classes from the cached dataset"""
        if 'categories' in self.coco_data:
            # Find max category_id
            max_id = max([cat['id'] for cat in self.coco_data['categories']])
            return max_id + 1
        return None
    
    def validate_num_classes(self):
        """Validate that num_classes matches the dataset"""
        actual_num_classes = self.get_num_classes_from_json()
        if actual_num_classes is not None and actual_num_classes != self.num_classes:
            print(f"WARNING: num_classes argument ({self.num_classes}) does not match "
                  f"actual dataset classes ({actual_num_classes}). "
                  f"Max category_id in dataset: {actual_num_classes - 1}")
            return False
        return True

    def load_image_and_target(self, idx):
        """Load image and target from cached data (no file I/O for JSON)"""
        # Get annotations for this image from cache
        annotations = self.annotations_by_image.get(idx, [])
        
        target = {}
        bboxes, classes, area, iscrowd = [], [], [], []
        
        for ann in annotations:
            bboxes.append(ann['bbox'])
            classes.append(ann['category_id'])
            area.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        # Convert to tensors
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        if len(bboxes) > 0:
            tmp = bboxes.reshape(-1, 2, 2)
            left_corner = tmp[:, 0]
            right_corner = tmp[:, 0] + tmp[:, 1]
            bboxes = torch.cat((left_corner, right_corner), dim=-1)
        
        classes = torch.as_tensor(classes, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd)
        
        target["boxes"] = bboxes
        target["labels"] = classes
        target["image_id"] = torch.tensor(idx)
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # Load image
        image_filename = self.image_id_to_filename[idx]
        image = Image.open(os.path.join(self.path, image_filename))
        w, h = image.size
        
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        return image, target

def build(image_set, args):
    path = Path(args.other_dataset_path)
    assert path.exists(), f'provided SF path {path} does not exist'
    
    # Pass num_classes from args if available
    num_classes = getattr(args, 'num_classes', 2)
    dataset = SmartFarm(path, image_set, transform=make_coco_transforms(image_set), num_classes=num_classes)
    
    # Validate num_classes on the first dataset (train)
    if image_set == 'train':
        dataset.validate_num_classes()
    
    return dataset

# if __name__ == '__main__':
#     MyDataset = SmartFarm('/Users/shijunshen/Documents/Code/dataset/Smart_Farm_Detection.v1i.coco')
#     print(MyDataset[0])