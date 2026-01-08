import os
import argparse
import torch
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config, VOC_CLASSES
from models import build_model
from datasets import VOCDataset, collate_fn, get_train_transforms, get_val_transforms
from utils.metrics import evaluate_detections
try:
    from torch.amp import GradScaler, autocast

except ImportError:
    from torch.cuda.amp import GradScaler, autocast


def get_lr_scheduler(optimizer, epochs, warmup_epochs=2):
    """Cosine annealing scheduler with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        optimizer.zero_grad()
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

        with autocast(device_type):
            losses = model(images, targets)
            loss = sum(losses.values())
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rpn_cls': f'{losses.get("rpn_cls_loss", 0):.3f}',
            'rpn_reg': f'{losses.get("rpn_reg_loss", 0):.3f}'
        })
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, device, num_classes):
    """Validate and compute mAP."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    for images, targets in tqdm(loader, desc='Validating'):
        images = images.to(device)
        detections = model(images)
        
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
    
    mAP, ap_per_class = evaluate_detections(
        all_predictions, all_targets, num_classes, iou_thresh=0.5
    )
    
    return mAP, ap_per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--max_iters', type=int, default=None, help='For testing')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_dataset = VOCDataset(
        args.data_root,
        image_set='trainval',
        transforms=get_train_transforms(Config.image_size)
    )
    
    val_dataset = VOCDataset(
        args.data_root,
        image_set='test',
        transforms=get_val_transforms(Config.image_size)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f'Train: {len(train_dataset)} images')
    print(f'Val: {len(val_dataset)} images')
    
    model = build_model(num_classes=Config.num_classes)
    model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay
    )
    
    scheduler = get_lr_scheduler(optimizer, args.epochs, Config.warmup_epochs)
    scaler = GradScaler()
    
    start_epoch = 0
    best_mAP = 0
    
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('best_mAP', 0)
        print(f'Resumed from epoch {start_epoch}')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch + 1, Config.grad_clip
        )
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}')
        
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            mAP, ap_per_class = validate(model, val_loader, device, Config.num_classes)
            print(f'mAP@0.5: {mAP:.4f}')
            
            for cls_idx, ap in ap_per_class.items():
                print(f'  {VOC_CLASSES[cls_idx - 1]}: {ap:.3f}')
            
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mAP': mAP,
                    'best_mAP': best_mAP
                }, os.path.join(args.output_dir, 'best.pth'))
                print(f'Saved best model with mAP: {best_mAP:.4f}')
        
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mAP': best_mAP
        }, os.path.join(args.output_dir, 'latest.pth'))
    
    print(f'\nTraining complete! Best mAP: {best_mAP:.4f}')


if __name__ == '__main__':
    main()
