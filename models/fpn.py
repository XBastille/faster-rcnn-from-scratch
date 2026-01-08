import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale detection."""
    
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: channels for C2, C3, C4, C5 (e.g., [256, 512, 1024, 2048])
            out_channels: output channels for all FPN levels
        """
        super().__init__()
        
        self.lateral_c5 = nn.Conv2d(in_channels_list[3], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.output_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        c2, c3, c4, c5 = features['c2'], features['c3'], features['c4'], features['c5']
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        p5 = self.output_p5(p5)
        p4 = self.output_p4(p4)
        p3 = self.output_p3(p3)
        p2 = self.output_p2(p2)
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)
        
        return {'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6}
