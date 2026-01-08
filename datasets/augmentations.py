import torchvision.transforms.functional as TF
import random


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    """Resize image and boxes to target size."""
    
    def __init__(self, size):
        self.size = size  # (H, W)
    
    def __call__(self, image, target):
        orig_w, orig_h = image.size
        new_h, new_w = self.size
        
        image = TF.resize(image, self.size)
        
        # Scale boxes
        if len(target['boxes']) > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            
            boxes = target['boxes']
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            target['boxes'] = boxes
        
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip."""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = TF.hflip(image)
            
            if len(target['boxes']) > 0:
                w = image.size[0]
                boxes = target['boxes']
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        return image, target


class ColorJitter:
    """Random color jittering."""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
    
    def __call__(self, image, target):
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = TF.adjust_brightness(image, factor)
        
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = TF.adjust_contrast(image, factor)
        
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            image = TF.adjust_saturation(image, factor)
        
        return image, target


class ToTensor:
    """Convert PIL Image to tensor and normalize."""
    
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        return image, target


def get_train_transforms(image_size=448):
    """Get training transforms."""
    return Compose([
        Resize((image_size, image_size)),
        RandomHorizontalFlip(prob=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ToTensor()
    ])


def get_val_transforms(image_size=448):
    """Get validation transforms."""
    return Compose([
        Resize((image_size, image_size)),
        ToTensor()
    ])
