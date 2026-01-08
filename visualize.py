import os
import argparse
import random
import torch
import cv2
from PIL import Image
from config import Config, VOC_CLASSES
from models import build_model
from datasets import VOCDataset, get_val_transforms

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
    (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0)
]


def draw_boxes(image, boxes, labels, scores, score_thresh=0.3):
    """Draw boxes on image."""

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[(label - 1) % len(COLORS)]
        class_name = VOC_CLASSES[label - 1]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{class_name}: {score:.2f}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 4), 
                     (x1 + text_size[0], y1), color, -1)
        
        cv2.putText(image, text, (x1, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


@torch.no_grad()
def generate_visualizations(checkpoint_path, data_root, output_dir, 
                            num_samples=20, score_thresh=0.3):
    """Generate sample prediction images."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=Config.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    dataset = VOCDataset(data_root, image_set='test', transforms=None)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    print(f"Generating {len(indices)} visualizations...")
    
    for i, idx in enumerate(indices):
        image_id = dataset.image_ids[idx]
        img_path = os.path.join(dataset.image_dir, f'{image_id}.jpg')
        orig_img = cv2.imread(img_path)
        orig_h, orig_w = orig_img.shape[:2]
        pil_img = Image.open(img_path).convert('RGB')
        transforms = get_val_transforms(Config.image_size)
        img_tensor, _ = transforms(pil_img, {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0)})
        img_tensor = img_tensor.unsqueeze(0).to(device)
        detections = model(img_tensor)[0]
        scale_x = orig_w / Config.image_size
        scale_y = orig_h / Config.image_size
        boxes = detections['boxes'].cpu().numpy()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        labels = detections['labels'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        vis_img = draw_boxes(orig_img.copy(), boxes, labels, scores, score_thresh)
        out_path = os.path.join(output_dir, f'{image_id}_pred.jpg')
        cv2.imwrite(out_path, vis_img)
        print(f"  [{i+1}/{len(indices)}] Saved: {out_path}")
    
    print(f"\nDone! Visualizations saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/best.pth')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs/visualizations')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--score_thresh', type=float, default=0.5)
    args = parser.parse_args()
    
    generate_visualizations(
        args.checkpoint, args.data_root, args.output_dir,
        args.num_samples, args.score_thresh
    )
