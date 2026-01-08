import torch
from utils.boxes import box_iou


def compute_ap(recall, precision):
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0

    for t in torch.linspace(0, 1, 11):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max().item()

    return ap / 11


def evaluate_detections(predictions, ground_truths, num_classes, iou_thresh=0.5):
    """
    Compute mAP for detection results.
    
    Args:
        predictions: list of dicts with 'boxes', 'labels', 'scores'
        ground_truths: list of dicts with 'boxes', 'labels'
        num_classes: number of classes (including background)
        iou_thresh: IoU threshold for matching
    
    Returns:
        mAP: mean Average Precision
        ap_per_class: dict of AP per class
    """
    ap_per_class = {}
    
    for cls_idx in range(1, num_classes):
        all_scores = []
        all_tp = []
        num_gt = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_mask = pred['labels'] == cls_idx
            pred_boxes = pred['boxes'][pred_mask]
            pred_scores = pred['scores'][pred_mask]
            gt_mask = gt['labels'] == cls_idx
            gt_boxes = gt['boxes'][gt_mask]
            num_gt += len(gt_boxes)
            if len(pred_boxes) == 0:
                continue
            
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            sorted_idx = pred_scores.argsort(descending=True)
            
            for idx in sorted_idx:
                score = pred_scores[idx]
                box = pred_boxes[idx:idx+1]
                if len(gt_boxes) == 0:
                    all_scores.append(score.item())
                    all_tp.append(False)
                    continue
                
                ious = box_iou(box, gt_boxes)[0]
                best_iou, best_idx = ious.max(dim=0)
                if best_iou >= iou_thresh and not gt_matched[best_idx]:
                    all_scores.append(score.item())
                    all_tp.append(True)
                    gt_matched[best_idx] = True

                else:
                    all_scores.append(score.item())
                    all_tp.append(False)
        
        if num_gt == 0:
            continue
        
        if len(all_scores) == 0:
            ap_per_class[cls_idx] = 0.0
            continue
        
        scores = torch.tensor(all_scores)
        tp = torch.tensor(all_tp)
        sorted_idx = scores.argsort(descending=True)
        tp = tp[sorted_idx]
        tp_cumsum = tp.cumsum(dim=0).float()
        fp_cumsum = (~tp).cumsum(dim=0).float()
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / num_gt
        ap = compute_ap(recall, precision)
        ap_per_class[cls_idx] = ap
    
    if len(ap_per_class) == 0:
        return 0.0, {}
    
    mAP = sum(ap_per_class.values()) / len(ap_per_class)
    return mAP, ap_per_class


def compute_map_range(predictions, ground_truths, num_classes,
                      iou_thresholds=None):
    """Compute mAP over multiple IoU thresholds (COCO-style)."""
    if iou_thresholds is None:
        iou_thresholds = torch.linspace(0.5, 0.95, 10)
    
    aps = []

    for thresh in iou_thresholds:
        mAP, _ = evaluate_detections(predictions, ground_truths, num_classes, thresh.item())
        aps.append(mAP)
    
    return sum(aps) / len(aps)


def evaluate_by_size(predictions, ground_truths, num_classes, iou_thresh=0.5):
    """
    Compute AP for small, medium, and large objects.
    
    Size definitions (COCO-style, based on box area):
    - Small: area < 32^2 = 1024 pixels
    - Medium: 32^2 <= area < 96^2 = 9216 pixels
    - Large: area >= 96^2 = 9216 pixels
    """
    def box_area(boxes):
        if len(boxes) == 0:
            return torch.tensor([])
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    def filter_by_size(preds, gts, min_area, max_area):
        """Filter predictions and ground truths by box area."""
        filtered_preds = []
        filtered_gts = []
        
        for pred, gt in zip(preds, gts):
            gt_areas = box_area(gt['boxes'])
            gt_mask = (gt_areas >= min_area) & (gt_areas < max_area)
            pred_areas = box_area(pred['boxes'])
            pred_mask = (pred_areas >= min_area) & (pred_areas < max_area)
            filtered_preds.append({
                'boxes': pred['boxes'][pred_mask],
                'labels': pred['labels'][pred_mask],
                'scores': pred['scores'][pred_mask]
            })

            filtered_gts.append({
                'boxes': gt['boxes'][gt_mask],
                'labels': gt['labels'][gt_mask]
            })
        
        return filtered_preds, filtered_gts
    
    small_max = 32 * 32
    medium_max = 96 * 96
    large_max = float('inf')
    small_preds, small_gts = filter_by_size(predictions, ground_truths, 0, small_max)
    medium_preds, medium_gts = filter_by_size(predictions, ground_truths, small_max, medium_max)
    large_preds, large_gts = filter_by_size(predictions, ground_truths, medium_max, large_max)
    ap_small, _ = evaluate_detections(small_preds, small_gts, num_classes, iou_thresh)
    ap_medium, _ = evaluate_detections(medium_preds, medium_gts, num_classes, iou_thresh)
    ap_large, _ = evaluate_detections(large_preds, large_gts, num_classes, iou_thresh)
    
    return {
        'AP_small': ap_small,
        'AP_medium': ap_medium,
        'AP_large': ap_large
    }

