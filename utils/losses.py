import torch
import torch.nn.functional as F


def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss (Huber loss)."""
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.sum()


def rpn_loss(objectness, box_deltas, labels, gt_deltas, 
             batch_size=256, pos_fraction=0.5):
    """
    Compute RPN losses.
    
    Args:
        objectness: (N,) predicted objectness scores (logits)
        box_deltas: (N, 4) predicted box deltas
        labels: (N,) anchor labels (1=pos, 0=neg, -1=ignore)
        gt_deltas: (N, 4) ground truth box deltas
        batch_size: number of anchors to sample
        pos_fraction: fraction of positive samples
    
    Returns:
        cls_loss: classification loss
        reg_loss: regression loss
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()
    num_pos_samples = min(int(batch_size * pos_fraction), num_pos)
    num_neg_samples = min(batch_size - num_pos_samples, num_neg)
    
    if num_pos > num_pos_samples and num_pos_samples > 0:
        pos_idx = torch.where(pos_mask)[0]
        perm = torch.randperm(len(pos_idx), device=labels.device)[:num_pos_samples]
        sampled_pos = pos_idx[perm]

    else:
        sampled_pos = torch.where(pos_mask)[0]
    
    if num_neg > num_neg_samples and num_neg_samples > 0:
        neg_idx = torch.where(neg_mask)[0]
        perm = torch.randperm(len(neg_idx), device=labels.device)[:num_neg_samples]
        sampled_neg = neg_idx[perm]

    else:
        sampled_neg = torch.where(neg_mask)[0]
    
    sampled_idx = torch.cat([sampled_pos, sampled_neg])
    sampled_labels = labels[sampled_idx]
    
    if len(sampled_idx) > 0:
        cls_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_idx],
            sampled_labels.float(),
            reduction='sum'
        ) / max(len(sampled_idx), 1)

    else:
        cls_loss = objectness.sum() * 0.0
    
    if len(sampled_pos) > 0:
        reg_loss = smooth_l1_loss(
            box_deltas[sampled_pos],
            gt_deltas[sampled_pos]
        ) / max(len(sampled_pos), 1)

    else:
        reg_loss = box_deltas.sum() * 0.0
    
    return cls_loss, reg_loss


def detection_loss(cls_logits, box_deltas, labels, gt_deltas):
    """
    Compute detection head losses.
    
    Args:
        cls_logits: (N, num_classes) class logits
        box_deltas: (N, num_classes * 4) box deltas per class
        labels: (N,) ground truth class labels
        gt_deltas: (N, 4) ground truth box deltas
    
    Returns:
        cls_loss: classification loss
        reg_loss: regression loss
    """
    num_classes = cls_logits.shape[1]
    cls_loss = F.cross_entropy(cls_logits, labels, reduction='mean')
    pos_mask = labels > 0
    
    if pos_mask.sum() > 0:
        pos_labels = labels[pos_mask]
        pos_box_deltas = box_deltas[pos_mask]
        pos_gt_deltas = gt_deltas[pos_mask]
        batch_idx = torch.arange(pos_mask.sum(), device=labels.device)
        pos_box_deltas = pos_box_deltas.view(-1, num_classes, 4)
        selected_deltas = pos_box_deltas[batch_idx, pos_labels]
        reg_loss = smooth_l1_loss(selected_deltas, pos_gt_deltas) / max(pos_mask.sum(), 1)
        
    else:
        reg_loss = box_deltas.sum() * 0.0
    
    return cls_loss, reg_loss
