import torch


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) tensor in (x1, y1, x2, y2) format
        boxes2: (M, 4) tensor in (x1, y1, x2, y2) format
    
    Returns:
        iou: (N, M) tensor of pairwise IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    
    return inter / (union + 1e-7)


def encode_boxes(gt_boxes, anchors):
    """
    Encode ground truth boxes relative to anchors.
    
    Args:
        gt_boxes: (N, 4) ground truth boxes (x1, y1, x2, y2)
        anchors: (N, 4) anchor boxes (x1, y1, x2, y2)
    
    Returns:
        deltas: (N, 4) encoded box deltas (dx, dy, dw, dh)
    """
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_h
    dx = (gt_cx - anchor_cx) / (anchor_w + 1e-7)
    dy = (gt_cy - anchor_cy) / (anchor_h + 1e-7)
    dw = torch.log(gt_w / (anchor_w + 1e-7) + 1e-7)
    dh = torch.log(gt_h / (anchor_h + 1e-7) + 1e-7)
    
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(deltas, anchors):
    """
    Decode box deltas to get predicted boxes.
    
    Args:
        deltas: (N, 4) encoded deltas (dx, dy, dw, dh)
        anchors: (N, 4) anchor boxes (x1, y1, x2, y2)
    
    Returns:
        boxes: (N, 4) decoded boxes (x1, y1, x2, y2)
    """
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h
    dw = torch.clamp(deltas[:, 2], max=4.0)
    dh = torch.clamp(deltas[:, 3], max=4.0)
    pred_cx = deltas[:, 0] * anchor_w + anchor_cx
    pred_cy = deltas[:, 1] * anchor_h + anchor_cy
    pred_w = torch.exp(dw) * anchor_w
    pred_h = torch.exp(dh) * anchor_h
    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h
    
    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes(boxes, img_size):
    """Clip boxes to image boundaries."""
    h, w = img_size
    x1 = boxes[:, 0].clamp(0, w)
    y1 = boxes[:, 1].clamp(0, h)
    x2 = boxes[:, 2].clamp(0, w)
    y2 = boxes[:, 3].clamp(0, h)
    return torch.stack([x1, y1, x2, y2], dim=1)



def remove_small_boxes(boxes, min_size=1):
    """Remove boxes smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return keep


def nms(boxes, scores, iou_threshold):
    """
    Non-maximum suppression.
    
    Args:
        boxes: (N, 4) boxes
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep: indices of boxes to keep
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, labels, iou_threshold):
    """NMS per class."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    max_coord = boxes.max()
    offsets = labels.to(boxes.dtype) * (max_coord + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    return nms(boxes_for_nms, scores, iou_threshold)
