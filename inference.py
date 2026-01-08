import argparse
import torch
import cv2
from PIL import Image
from config import Config, VOC_CLASSES
from models import build_model
from datasets.augmentations import Resize, ToTensor, Compose

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
    (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0)
]

def load_model(checkpoint_path, device):
    """Load trained model."""
    model = build_model(num_classes=Config.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, image_size):
    """Preprocess image for inference."""
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size  # (W, H)
    transforms = Compose([
        Resize((image_size, image_size)),
        ToTensor()
    ])
    
    target = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
    image_tensor, _ = transforms(image, target)
    
    return image_tensor, orig_size


def draw_detections(image, boxes, labels, scores, orig_size, 
                    image_size, score_thresh=0.3):
    """Draw bounding boxes on image."""
    scale_x = orig_size[0] / image_size
    scale_y = orig_size[1] / image_size
    
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)
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
def run_inference(model, image_path, device, score_thresh=0.3):
    """Run inference on single image."""
    image_tensor, orig_size = preprocess_image(image_path, Config.image_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    detections = model(image_tensor)[0]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    image = draw_detections(
        image, boxes, labels, scores, 
        orig_size, Config.image_size, score_thresh
    )
    
    return image, detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.jpg')
    parser.add_argument('--score_thresh', type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = load_model(args.checkpoint, device)
    print('Model loaded')
    
    image, detections = run_inference(
        model, args.image, device, args.score_thresh
    )
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, image)
    print(f'Saved result to {args.output}')
    
    print(f'\nDetections (score >= {args.score_thresh}):')
    for box, label, score in zip(
        detections['boxes'].cpu().numpy(),
        detections['labels'].cpu().numpy(),
        detections['scores'].cpu().numpy()
    ):
        if score >= args.score_thresh:
            class_name = VOC_CLASSES[label - 1]
            print(f'  {class_name}: {score:.3f} @ [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]')


if __name__ == '__main__':
    main()
