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


class SimpleMetricsCallback:
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_ious = []
        self.steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
            if 'eval_mean_iou' in logs:
                self.eval_ious.append(logs['eval_mean_iou'])
    
    def plot_metrics(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.steps[:len(self.train_losses)], self.train_losses, 'b-', label='Train Loss')
        if self.eval_losses:
            plt.plot(self.steps[:len(self.eval_losses)], self.eval_losses, 'r-', label='Eval Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if self.eval_ious:
            plt.plot(self.steps[:len(self.eval_ious)], self.eval_ious, 'g-', label='Mean IoU')
            plt.xlabel('Steps')
            plt.ylabel('Mean IoU')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('C:/Users/sorvi/Downloads/training_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()