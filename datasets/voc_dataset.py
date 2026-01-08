import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

from config import VOC_CLASSES


class VOCDataset(Dataset):
    """PASCAL VOC 2007 detection dataset."""
    
    def __init__(self, root, image_set='trainval', transforms=None):
        """
        Args:
            root: path to VOCdevkit folder
            image_set: 'trainval' or 'test'
            transforms: data augmentation transforms
        """
        self.root = root
        self.transforms = transforms
        self.image_set = image_set
        self.mosaic_prob = 0.5 if image_set == 'trainval' else 0.0
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(VOC_CLASSES)}
        self.idx_to_class = {idx + 1: cls for idx, cls in enumerate(VOC_CLASSES)}
        voc_root = os.path.join(root, 'VOCdevkit', 'VOC2007')
        split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{image_set}.txt')
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        if self.mosaic_prob > 0 and torch.rand(1).item() < self.mosaic_prob:
            image, target = self.load_mosaic(idx)
        else:
            image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
            image = Image.open(image_path).convert('RGB')
            
            annotation_path = os.path.join(self.annotation_dir, f'{image_id}.xml')
            boxes, labels = self._parse_annotation(annotation_path)
            
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long),
                'image_id': image_id
            }
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def load_mosaic(self, index):
        """Loads 4 images and stitches them into a mosaic (2x2 grid)."""
        indices = [index] + [torch.randint(0, len(self.image_ids), (1,)).item() for _ in range(3)]
        s = 448
        h_s, w_s = s // 2, s // 2
        mosaic_img = Image.new('RGB', (s, s), (114, 114, 114))
        mosaic_boxes = []
        mosaic_labels = []
        
        offsets = [(0, 0), (w_s, 0), (0, h_s), (w_s, h_s)]
        
        for i, idx in enumerate(indices):
            image_id = self.image_ids[idx]
            img_path = os.path.join(self.image_dir, f'{image_id}.jpg')
            img = Image.open(img_path).convert('RGB')
            ann_path = os.path.join(self.annotation_dir, f'{image_id}.xml')
            boxes, labels = self._parse_annotation(ann_path)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            
            img = img.resize((w_s, h_s))
            
            if len(boxes) > 0:
                orig_w, orig_h = Image.open(img_path).size
                
                scale_x = w_s / orig_w
                scale_y = h_s / orig_h
                
                boxes[:, 0] *= scale_x
                boxes[:, 2] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 3] *= scale_y
                
                x_off, y_off = offsets[i]
                boxes[:, 0] += x_off
                boxes[:, 2] += x_off
                boxes[:, 1] += y_off
                boxes[:, 3] += y_off
                
                mosaic_boxes.append(boxes)
                mosaic_labels.extend(labels)
            
            mosaic_img.paste(img, offsets[i])
        
        if len(mosaic_boxes) > 0:
            mosaic_boxes = torch.cat(mosaic_boxes, dim=0)
            mosaic_labels = torch.tensor(mosaic_labels, dtype=torch.long)
            mosaic_boxes[:, 0] = mosaic_boxes[:, 0].clamp(0, s)
            mosaic_boxes[:, 1] = mosaic_boxes[:, 1].clamp(0, s)
            mosaic_boxes[:, 2] = mosaic_boxes[:, 2].clamp(0, s)
            mosaic_boxes[:, 3] = mosaic_boxes[:, 3].clamp(0, s)

        else:
            mosaic_boxes = torch.zeros((0, 4), dtype=torch.float32)
            mosaic_labels = torch.zeros((0,), dtype=torch.long)
            
        target = {
            'boxes': mosaic_boxes,
            'labels': mosaic_labels,
            'image_id': f'mosaic_{index}'
        }
        
        return mosaic_img, target
    
    def _parse_annotation(self, annotation_path):
        """Parse VOC XML annotation."""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            if name not in self.class_to_idx:
                continue
            
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            boxes.append([xmin - 1, ymin - 1, xmax - 1, ymax - 1])
            labels.append(self.class_to_idx[name])
        
        return boxes, labels


def collate_fn(batch):
    """Custom collate function for detection."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    return images, targets
