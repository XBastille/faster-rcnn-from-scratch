import torch

class Config:
    data_root = './data'
    num_classes = 21  
    backbone_out_channels = [256, 512, 1024, 2048]  # C2, C3, C4, C5
    fpn_channels = 256
    anchor_sizes = [32, 64, 128, 256, 512]
    anchor_ratios = [0.5, 1.0, 2.0]
    rpn_pre_nms_top_n_train = 2000
    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_train = 1000
    rpn_post_nms_top_n_test = 300
    rpn_nms_thresh = 0.7
    rpn_pos_thresh = 0.7
    rpn_neg_thresh = 0.3
    rpn_batch_size = 256
    rpn_pos_fraction = 0.5
    roi_pool_size = 7
    roi_batch_size = 128
    roi_pos_fraction = 0.25
    roi_pos_thresh = 0.5
    roi_neg_thresh_hi = 0.5
    roi_neg_thresh_lo = 0.0
    image_size = 448
    batch_size = 16
    epochs = 80
    lr = 0.04  
    momentum = 0.9
    weight_decay = 0.0005
    warmup_epochs = 2
    grad_clip = 1.0
    score_thresh = 0.05
    nms_thresh = 0.5
    max_detections = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = './outputs'


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
