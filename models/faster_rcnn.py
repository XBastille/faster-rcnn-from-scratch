import torch
import torch.nn as nn
from .backbone import ResNet50
from .fpn import FPN
from .rpn import RPN
from .roi_head import ROIHead


class FasterRCNN(nn.Module):
    """Faster R-CNN with FPN backbone."""
    
    def __init__(self, num_classes=21):
        super().__init__()
        
        self.backbone = ResNet50()
        
        in_channels = [256, 512, 1024, 2048]  # C2, C3, C4, C5 channels
        self.fpn = FPN(in_channels, out_channels=256)
        
        self.rpn = RPN(in_channels=256)
        
        self.roi_head = ROIHead(in_channels=256, num_classes=num_classes)
    
    def forward(self, images, targets=None):
        """
        Args:
            images: (B, 3, H, W) tensor or list of (3, H, W) tensors
            targets: list of dicts with 'boxes' and 'labels' (training only)
        
        Returns:
            Training: dict of losses
            Inference: list of dicts with 'boxes', 'labels', 'scores'
        """
        if isinstance(images, list):
            images = torch.stack(images)
        
        image_size = (images.shape[2], images.shape[3])
        
        backbone_features = self.backbone(images)
        fpn_features = self.fpn(backbone_features)
        
        proposals, rpn_losses = self.rpn(fpn_features, image_size, targets)
        
        detections, det_losses = self.roi_head(
            fpn_features, proposals, image_size, targets
        )
        
        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(det_losses)
            return losses
        
        return detections


def build_model(num_classes=21):
    """Build Faster R-CNN model."""
    return FasterRCNN(num_classes=num_classes)
