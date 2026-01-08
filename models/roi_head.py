import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from utils.boxes import box_iou, encode_boxes, decode_boxes, batched_nms
from utils.losses import detection_loss


class ROIHead(nn.Module):
    """Fast R-CNN detection head with ROI pooling."""
    
    def __init__(self, in_channels=256, num_classes=21, pool_size=7):
        super().__init__()
        
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        self.batch_size = 128
        self.pos_fraction = 0.25
        self.pos_thresh = 0.5
        self.neg_thresh_hi = 0.5
        self.neg_thresh_lo = 0.0
        self.score_thresh = 0.05
        self.nms_thresh = 0.5
        self.max_detections = 100
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
    
    def forward(self, features, proposals, image_size, targets=None):
        """
        Args:
            features: dict of FPN features
            proposals: list of (N, 4) proposal tensors per image
            image_size: (H, W) of input image
            targets: list of dicts with 'boxes' and 'labels' (training only)
        
        Returns:
            detections: list of dicts with 'boxes', 'labels', 'scores'
            losses: dict of detection losses (training only)
        """
        device = proposals[0].device
        
        if self.training and targets is not None:
            proposals, labels, gt_deltas = self._sample_proposals(
                proposals, targets, device
            )
        
        pooled_features = self._roi_pool(features, proposals, image_size)
        x = pooled_features.view(pooled_features.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_logits = self.cls_score(x)
        box_deltas = self.bbox_pred(x)
        losses = {}
        
        if self.training and targets is not None:
            cls_loss, reg_loss = detection_loss(
                cls_logits, box_deltas, labels, gt_deltas
            )

            losses = {
                'det_cls_loss': cls_loss,
                'det_reg_loss': reg_loss
            }
        
        detections = self._post_process(
            cls_logits, box_deltas, proposals, image_size
        )
        
        return detections, losses
    
    def _roi_pool(self, features, proposals, image_size):
        """Pool features from FPN levels based on proposal size."""
        all_pooled = []
        feature_map = features['p3']
        spatial_scale = 1.0 / 8.0
        
        for i, props in enumerate(proposals):
            if len(props) == 0:
                continue
            
            batch_idx = torch.full((len(props), 1), i, device=props.device, dtype=props.dtype)
            rois = torch.cat([batch_idx, props], dim=1)
            pooled = roi_align(
                feature_map,
                rois,
                output_size=self.pool_size,
                spatial_scale=spatial_scale,
                aligned=True
            )

            all_pooled.append(pooled)
        
        if len(all_pooled) == 0:
            return torch.zeros(0, 256, self.pool_size, self.pool_size, device=feature_map.device)
        
        return torch.cat(all_pooled, dim=0)
    
    def _sample_proposals(self, proposals, targets, device):
        """Sample proposals for training."""
        sampled_proposals = []
        sampled_labels = []
        sampled_gt_deltas = []
        
        for props, target in zip(proposals, targets):
            gt_boxes = target['boxes'].to(device)
            gt_labels = target['labels'].to(device)
            if len(props) == 0:
                continue
            
            iou = box_iou(props, gt_boxes)
            max_iou, matched_idx = iou.max(dim=1)
            labels = torch.zeros(len(props), dtype=torch.long, device=device)
            pos_mask = max_iou >= self.pos_thresh
            labels[pos_mask] = gt_labels[matched_idx[pos_mask]]
            neg_mask = (max_iou >= self.neg_thresh_lo) & (max_iou < self.neg_thresh_hi)
            num_pos = int(self.batch_size * self.pos_fraction)
            pos_idx = torch.where(pos_mask)[0]
            neg_idx = torch.where(neg_mask)[0]
            num_pos = min(len(pos_idx), num_pos)
            num_neg = min(len(neg_idx), self.batch_size - num_pos)
            if len(pos_idx) > num_pos:
                perm = torch.randperm(len(pos_idx), device=device)[:num_pos]
                pos_idx = pos_idx[perm]
            
            if len(neg_idx) > num_neg:
                perm = torch.randperm(len(neg_idx), device=device)[:num_neg]
                neg_idx = neg_idx[perm]
            
            sampled_idx = torch.cat([pos_idx, neg_idx])
            sampled_proposals.append(props[sampled_idx])
            sampled_labels.append(labels[sampled_idx])
            gt_deltas = torch.zeros(len(sampled_idx), 4, device=device)
            pos_in_sampled = torch.arange(len(pos_idx), device=device)
            
            if len(pos_idx) > 0:
                matched_gt = gt_boxes[matched_idx[pos_idx]]
                gt_deltas[pos_in_sampled] = encode_boxes(matched_gt, props[pos_idx])
            
            sampled_gt_deltas.append(gt_deltas)
        
        return sampled_proposals, torch.cat(sampled_labels), torch.cat(sampled_gt_deltas)
    
    def _post_process(self, cls_logits, box_deltas, proposals, image_size):
        """Post-process predictions to get final detections."""
        device = cls_logits.device
        scores = F.softmax(cls_logits, dim=1)
        all_proposals = torch.cat(proposals, dim=0)
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for cls_idx in range(1, self.num_classes):
            cls_scores = scores[:, cls_idx]
            box_idx = cls_idx * 4
            cls_deltas = box_deltas[:, box_idx:box_idx + 4]
            boxes = decode_boxes(cls_deltas, all_proposals)
            mask = cls_scores > self.score_thresh
            boxes = boxes[mask]
            cls_scores = cls_scores[mask]
            all_boxes.append(boxes)
            all_scores.append(cls_scores)
            all_labels.append(torch.full((len(boxes),), cls_idx, device=device, dtype=torch.long))
        
        if len(all_boxes) == 0:
            return [{'boxes': torch.zeros(0, 4, device=device),
                     'labels': torch.zeros(0, dtype=torch.long, device=device),
                     'scores': torch.zeros(0, device=device)}]
        
        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        
        keep = batched_nms(boxes, scores, labels, self.nms_thresh)
        keep = keep[:self.max_detections]
        
        return [{
            'boxes': boxes[keep],
            'labels': labels[keep],
            'scores': scores[keep]
        }]
