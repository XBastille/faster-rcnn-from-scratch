import argparse
import torch
import torch.onnx
import os
from config import Config
from models import build_model

class FasterRCNNWrapper(torch.nn.Module):
    """Wrapper for ONNX export (inference mode only)."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        self.model.eval()
        features = self.model.backbone(x)
        fpn_features = self.model.fpn(features)
        image_size = (x.shape[2], x.shape[3])
        proposals, _ = self.model.rpn(fpn_features, image_size, None)
        detections, _ = self.model.roi_head(fpn_features, proposals, image_size, None)
        if len(detections) > 0:
            d = detections[0]
            return d['boxes'], d['labels'], d['scores']
        
        else:
            return (torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), 
                    torch.zeros(0))


def export_onnx(checkpoint_path, output_path, image_size=448):
    """Export model to ONNX."""
    device = torch.device('cpu') 
    model = build_model(num_classes=Config.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    wrapper = FasterRCNNWrapper(model)
    dummy_input = torch.randn(1, 3, image_size, image_size)
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=['image'],
        output_names=['boxes', 'labels', 'scores'],
        dynamic_axes={
            'image': {0: 'batch'},
            'boxes': {0: 'num_detections'},
            'labels': {0: 'num_detections'},
            'scores': {0: 'num_detections'}
        }
    )
    
    print(f"Exported successfully!")
    print(f"File size: {os.path.getsize(output_path) / 1e6:.2f} MB")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/best.pth')
    parser.add_argument('--output', type=str, default='outputs/model.onnx')
    args = parser.parse_args()
    
    export_onnx(args.checkpoint, args.output)
