import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config, VOC_CLASSES
from models import build_model
from datasets import VOCDataset, collate_fn, get_val_transforms
from utils.metrics import evaluate_detections, compute_map_range

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Full evaluation with multiple IoU thresholds."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_time = 0
    num_images = 0
    
    for images, targets in tqdm(loader, desc='Evaluating'):
        images = images.to(device)
        start = time.time()
        detections = model(images)
        total_time += time.time() - start
        num_images += len(images)
        
        for det, target in zip(detections, targets):
            all_predictions.append({
                'boxes': det['boxes'].cpu(),
                'labels': det['labels'].cpu(),
                'scores': det['scores'].cpu()
            })
            all_targets.append({
                'boxes': target['boxes'],
                'labels': target['labels']
            })
    
    mAP50, ap_per_class = evaluate_detections(
        all_predictions, all_targets, num_classes, iou_thresh=0.5
    )
    
    mAP75, _ = evaluate_detections(
        all_predictions, all_targets, num_classes, iou_thresh=0.75
    )
    
    mAP_coco = compute_map_range(all_predictions, all_targets, num_classes)
    fps = num_images / total_time
    
    return {
        'mAP@0.5': mAP50,
        'mAP@0.75': mAP75,
        'mAP@0.5:0.95': mAP_coco,
        'FPS': fps,
        'ap_per_class': ap_per_class
    }


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    test_dataset = VOCDataset(
        args.data_root,
        image_set='test',
        transforms=get_val_transforms(Config.image_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f'Test: {len(test_dataset)} images')
    model = build_model(num_classes=Config.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "?")}')
    print(f'Parameters: {count_parameters(model):.2f}M')
    results = evaluate(model, test_loader, device, Config.num_classes)
    print('\n' + '=' * 50)
    print('EVALUATION RESULTS')
    print('=' * 50)
    print(f'mAP@0.5:      {results["mAP@0.5"]:.4f}')
    print(f'mAP@0.75:     {results["mAP@0.75"]:.4f}')
    print(f'mAP@0.5:0.95: {results["mAP@0.5:0.95"]:.4f}')
    print(f'FPS:          {results["FPS"]:.1f}')
    print('=' * 50)
    print('\nPer-class AP@0.5:')
    for cls_idx, ap in sorted(results['ap_per_class'].items()):
        print(f'  {VOC_CLASSES[cls_idx - 1]:15s}: {ap:.3f}')


if __name__ == '__main__':
    main()
