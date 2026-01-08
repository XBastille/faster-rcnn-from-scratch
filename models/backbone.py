import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.GroupNorm(32, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.GroupNorm(32, out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50 backbone returning multi-scale features C2-C5."""
    
    def __init__(self):
        super().__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 3, stride=1)   
        self.layer2 = self._make_layer(128, 4, stride=2) 
        self.layer3 = self._make_layer(256, 6, stride=2)  
        self.layer4 = self._make_layer(512, 3, stride=2)  
        
        self._init_weights()
    
    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                         1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels * Bottleneck.expansion)
            )
        
        layers = [Bottleneck(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * Bottleneck.expansion
        
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        c2 = self.layer1(x)  
        c3 = self.layer2(c2)  
        c4 = self.layer3(c3)  
        c5 = self.layer4(c4) 
        
        return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}
