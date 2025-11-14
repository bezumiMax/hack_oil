import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, input_dir, target_dir, processor, size=(512, 512), gray_mapping=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.processor = processor
        self.size = size
        self.gray_mapping = gray_mapping
        self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
            

class SegFormerWithResize(nn.Module):
    def __init__(self, original_model, num_new_classes=40, class_names=None, gray_mapping=None):
        super().__init__()
        self.original_model = original_model
        self.num_new_classes = num_new_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_new_classes)]
        self.gray_mapping = gray_mapping
        
        self.replace_classifier()
        
    def replace_classifier(self):
        in_channels = self.original_model.decode_head.classifier.in_channels
        new_classifier = nn.Conv2d(in_channels, self.num_new_classes, kernel_size=1)
        
        nn.init.normal_(new_classifier.weight, mean=0.0, std=0.02)
        if new_classifier.bias is not None:
            nn.init.zeros_(new_classifier.bias)
        
        self.original_model.decode_head.classifier = new_classifier
        self.original_model.config.num_labels = self.num_new_classes
        
    def forward(self, pixel_values, labels=None):
        batch_size = pixel_values.shape[0]
        original_size = pixel_values.shape[2:]
        
        # Уменьшаем вход: 640×640 → 512×512
        resized_input = torch.nn.functional.interpolate(
            pixel_values, 
            size=(512, 512), 
            mode='bilinear', 
            align_corners=False
        )
        
        outputs = self.original_model(pixel_values=resized_input, labels=labels)
        
        # Увеличиваем выход обратно: 512×512 → 640×640
        outputs.logits = torch.nn.functional.interpolate(
            outputs.logits, 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return outputs