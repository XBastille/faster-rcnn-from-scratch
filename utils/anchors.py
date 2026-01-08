import torch
import math
from .boxes import box_iou

class AnchorGenerator:
    """Generate anchors for all FPN levels."""
    
    def __init__(self, sizes=(32, 64, 128, 256, 512), ratios=(0.5, 1.0, 2.0)):
        """
        Args:
            sizes: base anchor sizes for each FPN level
            ratios: aspect ratios for anchors
        """
        self.sizes = sizes
        self.ratios = ratios
        self.num_anchors = len(ratios)
        
        self.base_anchors = self._generate_base_anchors()
    
    def _generate_base_anchors(self):
        """Generate base anchor templates centered at (0, 0)."""
        base_anchors = []
        
        for size in self.sizes:
            anchors_for_level = []
            for ratio in self.ratios:
                h = size
                w = size
                h = size / math.sqrt(ratio)
                w = size * math.sqrt(ratio)
                
                anchors_for_level.append([-w/2, -h/2, w/2, h/2])
            
            base_anchors.append(torch.tensor(anchors_for_level, dtype=torch.float32))
        
        return base_anchors
    
    def generate_anchors(self, feature_maps, image_size, device):
        """
        Generate anchors for all FPN levels.
        
        Args:
            feature_maps: dict with 'p2', 'p3', 'p4', 'p5' feature maps
            image_size: (H, W) of input image
            device: torch device
        
        Returns:
            all_anchors: (N, 4) tensor of all anchors
            anchors_per_level: list of number of anchors per level
        """
        all_anchors = []
        anchors_per_level = []
        
        strides = [4, 8, 16, 32, 64]
        level_keys = ['p2', 'p3', 'p4', 'p5', 'p6']
        
        for idx, (key, stride) in enumerate(zip(level_keys, strides)):
            feat = feature_maps[key]
            h, w = feat.shape[-2:]
            base_anchors = self.base_anchors[idx].to(device)
            shift_x = (torch.arange(w, device=device) + 0.5) * stride
            shift_y = (torch.arange(h, device=device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            anchors = shifts[:, None, :] + base_anchors[None, :, :]
            anchors = anchors.reshape(-1, 4)
            all_anchors.append(anchors)
            anchors_per_level.append(len(anchors))
        
        return torch.cat(all_anchors, dim=0), anchors_per_level


def match_anchors_to_gt(anchors, gt_boxes, pos_thresh=0.7, neg_thresh=0.3):
    """
    Match anchors to ground truth boxes.
    
    Args:
        anchors: (N, 4) anchor boxes
        gt_boxes: (M, 4) ground truth boxes
        pos_thresh: IoU threshold for positive anchors
        neg_thresh: IoU threshold for negative anchors
    
    Returns:
        matched_gt_idx: (N,) index of matched gt box for each anchor (-1 for negative)
        labels: (N,) 1 for positive, 0 for negative, -1 for ignore
    """
    
    num_anchors = anchors.shape[0]
    device = anchors.device
    
    if gt_boxes.shape[0] == 0:
        return (torch.full((num_anchors,), -1, dtype=torch.long, device=device),
                torch.zeros(num_anchors, dtype=torch.long, device=device))
    
    iou = box_iou(anchors, gt_boxes)  # (N, M)
    max_iou, matched_gt_idx = iou.max(dim=1)
    labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    labels[max_iou < neg_thresh] = 0
    labels[max_iou >= pos_thresh] = 1
    gt_max_iou, gt_argmax = iou.max(dim=0)
    labels[gt_argmax] = 1
    matched_gt_idx[gt_argmax] = torch.arange(len(gt_boxes), device=device)
    
    return matched_gt_idx, labels
