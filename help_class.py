import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class SegmentationDataset(Dataset):
    def __init__(self, input_dir, target_dir, processor, size=(640, 640), gray_mapping=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.processor = processor
        self.size = size
        self.gray_mapping = gray_mapping or {}

        self.image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png'))])
        self.mask_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png'))])
    
    def __len__(self):
        return len(self.image_files)
    
    # не используется, но варнинги требуют
    def __getitem__(self, idx):
        image_path = os.path.join(self.input_dir, self.image_files[idx])
        mask_path = os.path.join(self.target_dir, self.mask_files[idx])
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        
        if self.gray_mapping:
            new_mask = np.zeros_like(mask_array, dtype=np.int64)
            for gray_val, class_id in self.gray_mapping.items():
                new_mask[mask_array == gray_val] = class_id
            mask_array = new_mask
        
        # Отключаем ресайз в processor
        inputs = self.processor(
            images=image, 
            segmentation_maps=mask_array, 
            return_tensors="pt",
            do_resize=False,
            do_normalize=True,
            do_rescale=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': inputs['labels'].squeeze().long()
        }
            

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

        # танцы с бубном
        resized_labels = torch.nn.functional.interpolate(
            labels.unsqueeze(1).float(),
            size=(512, 512), 
            mode='nearest',
        ).squeeze(1).long()

        # Уменьшаем вход: 640×640 → 512×512
        resized_input = torch.nn.functional.interpolate(
            pixel_values, 
            size=(512, 512), 
            mode='bilinear', 
            align_corners=False
        )

        print(resized_input.shape)
        print(resized_labels.shape)
        
        outputs = self.original_model(
            pixel_values=resized_input, 
            labels=resized_labels
        )
        
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
        self.steps = []
        self.train_losses = []
        self.eval_losses = []
        self.eval_ious = []
    
    def on_init_end(self, args, state, control, **kwargs):
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        pass
    
    def on_train_end(self, args, state, control, **kwargs):
        pass
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        pass
    
    def on_epoch_end(self, args, state, control, **kwargs):
        pass
    
    def on_step_begin(self, args, state, control, **kwargs):
        pass
    
    def on_step_end(self, args, state, control, **kwargs):
        pass
    
    def on_evaluate(self, args, state, control, **kwargs):
        pass
    
    def on_prediction_step(self, args, state, control, **kwargs):
        pass
    
    # ДОБАВЛЯЕМ НЕДОСТАЮЩИЕ МЕТОДЫ:
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Вызывается перед шагом оптимизатора"""
        return control
    
    def on_optimizer_step(self, args, state, control, **kwargs):
        """Вызывается после шага оптимизатора"""
        return control
    
    def on_save(self, args, state, control, **kwargs):
        """Вызывается при сохранении модели"""
        return control
    
    def on_substep_end(self, args, state, control, **kwargs):
        """Вызывается в конце подшага (при gradient_accumulation_steps)"""
        return control
    
    def on_pre_save(self, args, state, control, **kwargs):
        """Вызывается перед сохранением модели"""
        return control
    
    def on_pre_evaluate(self, args, state, control, **kwargs):
        """Вызывается перед оценкой"""
        return control
    
    def on_pre_prediction_step(self, args, state, control, **kwargs):
        """Вызывается перед шагом предсказания"""
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при логировании - здесь собираем метрики"""
        if logs is not None:
            if 'loss' in logs and 'eval_loss' not in logs:
                self.steps.append(state.global_step)
                self.train_losses.append(logs['loss'])
            elif 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                if 'eval_mean_iou' in logs:
                    self.eval_ious.append(logs['eval_mean_iou'])
    
    def plot_metrics(self):
        """Рисует графики метрик"""
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
        plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()