import torch
import torch.nn as nn
from utils.boxes import decode_boxes, encode_boxes, clip_boxes, remove_small_boxes, nms
from utils.anchors import AnchorGenerator, match_anchors_to_gt
from utils.losses import rpn_loss


class RPNHead(nn.Module):
    """RPN prediction head."""
    
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: dict of FPN features
        
        Returns:
            objectness: list of (B, A, H, W) tensors
            box_deltas: list of (B, A*4, H, W) tensors
        """
        objectness = []
        box_deltas = []
        
        for key in ['p2', 'p3', 'p4', 'p5', 'p6']:
            feat = features[key]
            x = torch.relu(self.conv(feat))
            objectness.append(self.cls_logits(x))
            box_deltas.append(self.bbox_pred(x))
        
        return objectness, box_deltas


class RPN(nn.Module):
    """Region Proposal Network."""
    
    def __init__(self, in_channels=256, anchor_sizes=(32, 64, 128, 256, 512),
                 anchor_ratios=(0.5, 1.0, 2.0)):
        super().__init__()
        
        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        num_anchors = len(anchor_ratios)
        
        self.head = RPNHead(in_channels, num_anchors)
        self.pre_nms_top_n_train = 2000
        self.pre_nms_top_n_test = 1000
        self.post_nms_top_n_train = 1000
        self.post_nms_top_n_test = 300
        self.nms_thresh = 0.7
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3
        self.batch_size = 256
        self.pos_fraction = 0.5
        self.min_size = 1
    
    def forward(self, features, image_size, targets=None):
        """
        Args:
            features: dict of FPN features
            image_size: (H, W) of input image
            targets: list of dicts with 'boxes' key (training only)
        
        Returns:
            proposals: (N, 4) tensor of proposals
            losses: dict of RPN losses (training only)
        """
        objectness, box_deltas = self.head(features)
        device = objectness[0].device
        batch_size = objectness[0].shape[0]
        anchors, anchors_per_level = self.anchor_generator.generate_anchors(
            features, image_size, device
        )
        
        objectness_flat = []
        box_deltas_flat = []
        
        for obj, delta in zip(objectness, box_deltas):
            B, A, H, W = obj.shape
            obj = obj.permute(0, 2, 3, 1).reshape(B, -1)
            delta = delta.permute(0, 2, 3, 1).reshape(B, -1, 4)
            objectness_flat.append(obj)
            box_deltas_flat.append(delta)
        
        objectness_flat = torch.cat(objectness_flat, dim=1)  # (B, N)
        box_deltas_flat = torch.cat(box_deltas_flat, dim=1)  # (B, N, 4)
        
        proposals = self._generate_proposals(
            anchors, objectness_flat, box_deltas_flat, image_size,
            self.training
        )
        
        losses = {}
        
        if self.training and targets is not None:
            losses = self._compute_loss(
                anchors, objectness_flat, box_deltas_flat, targets
            )
        
        return proposals, losses
    
    def _generate_proposals(self, anchors, objectness, box_deltas, image_size, training):
        """Generate proposals from RPN predictions."""
        device = anchors.device
        batch_size = objectness.shape[0]
        pre_nms_top_n = self.pre_nms_top_n_train if training else self.pre_nms_top_n_test
        post_nms_top_n = self.post_nms_top_n_train if training else self.post_nms_top_n_test
        all_proposals = []
        
        for i in range(batch_size):
            scores = objectness[i].sigmoid()
            deltas = box_deltas[i]
            proposals = decode_boxes(deltas, anchors)
            proposals = proposals.detach()
            proposals = clip_boxes(proposals, image_size)
            keep = remove_small_boxes(proposals, self.min_size)
            proposals = proposals[keep]
            scores = scores[keep]
            if len(scores) > pre_nms_top_n:
                _, top_idx = scores.topk(pre_nms_top_n)
                proposals = proposals[top_idx]
                scores = scores[top_idx]
            
            keep = nms(proposals, scores, self.nms_thresh)
            keep = keep[:post_nms_top_n]
            proposals = proposals[keep]
            all_proposals.append(proposals)
        
        return all_proposals
    
    def _compute_loss(self, anchors, objectness, box_deltas, targets):
        """Compute RPN loss."""
        device = anchors.device
        batch_size = objectness.shape[0]
        total_cls_loss = 0
        total_reg_loss = 0
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            
            matched_gt_idx, labels = match_anchors_to_gt(
                anchors, gt_boxes, self.pos_thresh, self.neg_thresh
            )
            
            pos_mask = labels == 1
            gt_deltas = torch.zeros_like(box_deltas[i])
            
            if pos_mask.sum() > 0:
                matched_gt_boxes = gt_boxes[matched_gt_idx[pos_mask]]
                gt_deltas[pos_mask] = encode_boxes(matched_gt_boxes, anchors[pos_mask]).to(gt_deltas.dtype)
            
            cls_loss, reg_loss = rpn_loss(
                objectness[i], box_deltas[i], labels, gt_deltas,
                self.batch_size, self.pos_fraction
            )
            
            total_cls_loss += cls_loss
            total_reg_loss += reg_loss
        
        return {
            'rpn_cls_loss': total_cls_loss / batch_size,
            'rpn_reg_loss': total_reg_loss / batch_size
        }
