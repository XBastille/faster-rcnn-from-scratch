import argparse
import torch
import os
from config import Config
from models import build_model


def export_torchscript(checkpoint_path, output_path, image_size=448):
    """Export model to TorchScript."""
    device = torch.device('cpu')
    
    model = build_model(num_classes=Config.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    print(f"Tracing model...")
    
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
        print(f"Exported (trace) to {output_path}")
    except Exception as e:
        print(f"Trace failed: {e}")
        print("Trying torch.jit.script instead...")
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
        print(f"Exported (script) to {output_path}")
    
    
    print(f"File size: {os.path.getsize(output_path) / 1e6:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/best.pth')
    parser.add_argument('--output', type=str, default='outputs/model.pt')
    args = parser.parse_args()
    
    export_torchscript(args.checkpoint, args.output)
