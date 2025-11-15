import sys
import os

# Добавляем путь к пользовательским пакетам
user_site_packages = os.path.expanduser("~/AppData/Roaming/Python/Python313/site-packages")
if user_site_packages not in sys.path:
    sys.path.insert(0, user_site_packages)

import torch
import requests
from PIL import Image
import os
from torch.utils.data import Dataset, ConcatDataset, Subset
from transformers import AutoProcessor, AutoModelForSemanticSegmentation
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from help_class import SegFormerWithResize, SegmentationDataset, SimpleMetricsCallback, MemoryCleanupCallback
from for_cross_entropy import CUSTOM_CLASS_NAMES, GRAY_TO_CLASS_MAPPING
import gc



metrics_callback = SimpleMetricsCallback()

def create_custom_segformer(model_name="nvidia/segformer-b0-finetuned-ade-512-512"):    
    processor = AutoProcessor.from_pretrained(model_name,
                                              do_resize=False,
                                            size=(512, 512))
    original_model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

    model = SegFormerWithResize(
        original_model=original_model,
        num_new_classes=len(CUSTOM_CLASS_NAMES),
        class_names=CUSTOM_CLASS_NAMES,
        gray_mapping=GRAY_TO_CLASS_MAPPING
    )    
    return processor, model

processor, model = create_custom_segformer()

train_input_dir = "C:/Users/sorvi/Downloads/dataset/train/input"
train_target_dir = "C:/Users/sorvi/Downloads/dataset/train/target"
val_input_dir = "C:/Users/sorvi/Downloads/dataset/val/input"
val_target_dir = "C:/Users/sorvi/Downloads/dataset/val/target"
train_input_dir1 = "C:/Users/sorvi/Downloads/dataset1/train/input"
train_target_dir1 = "C:/Users/sorvi/Downloads/dataset1/train/target"
train_input_dir2 = "C:/Users/sorvi/Downloads/dataset2/train/input"
train_target_dir2 = "C:/Users/sorvi/Downloads/dataset2/train/target"
train_input_dir3 = "C:/Users/sorvi/Downloads/dataset3/train/input"
train_target_dir3 = "C:/Users/sorvi/Downloads/dataset3/train/target"
train_input_dir4 = "C:/Users/sorvi/Downloads/dataset4/train/input"
train_target_dir4 = "C:/Users/sorvi/Downloads/dataset4/train/target"


train_dataset = SegmentationDataset(
    input_dir=train_input_dir,
    target_dir=train_target_dir,
    processor=processor,
    size=(640, 640),
    gray_mapping=GRAY_TO_CLASS_MAPPING
)

train_dataset1 = SegmentationDataset(
    input_dir=train_input_dir1,
    target_dir=train_target_dir1,
    processor=processor,
    size=(640, 640),
    gray_mapping=GRAY_TO_CLASS_MAPPING
)

train_dataset2 = SegmentationDataset(
    input_dir=train_input_dir2,
    target_dir=train_target_dir2,
    processor=processor,
    size=(640, 640),
    gray_mapping=GRAY_TO_CLASS_MAPPING
)

'''
train_dataset3 = SegmentationDataset(
    input_dir=train_input_dir3,
    target_dir=train_target_dir3,
    processor=processor,
    size=(640, 640),
    gray_mapping=GRAY_TO_CLASS_MAPPING
)

train_dataset4 = SegmentationDataset(
    input_dir=train_input_dir4,
    target_dir=train_target_dir4,
    processor=processor,
    size=(640, 640),
    gray_mapping=GRAY_TO_CLASS_MAPPING
)
'''

val_dataset = SegmentationDataset(
    input_dir=val_input_dir,
    target_dir=val_target_dir,
    processor=processor,
    size=(640, 640),
    gray_mapping=GRAY_TO_CLASS_MAPPING
)

val_indices = list(range(0, len(val_dataset), 100))  
small_val_dataset = Subset(val_dataset, val_indices)

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    logits_tensor = torch.from_numpy(logits)
    predictions = torch.argmax(logits_tensor, dim=1)

    metrics = metric._compute(
        predictions=predictions.numpy(),
        references=labels,
        num_labels=model.original_model.config.num_labels,
        ignore_index=255
    )

    return {
        "eval_loss": float(metrics.get("mean_iou", 0.0)),
        "eval_mean_iou": float(metrics.get("mean_iou", 0.0)),
        "eval_mean_accuracy": float(metrics.get("mean_accuracy", 0.0)),
        "eval_overall_accuracy": float(metrics.get("overall_accuracy", 0.0))
    }

training_args = TrainingArguments(
    output_dir="C:/Users/sorvi/Downloads/segmentation-finetuned",
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    save_steps=10000,
    eval_steps=10000,
    logging_steps=10,
    logging_dir="./logs",
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    remove_unused_columns=False, 
)

combined_dataset = ConcatDataset([
    train_dataset,
    train_dataset1,
    train_dataset2
    #train_dataset3,
    #train_dataset4
])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
    eval_dataset=small_val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback, MemoryCleanupCallback()],
)

print("Начинаем обучение...")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
training_args.disable_tqdm = False
training_args.logging_steps = 10
trainer.train()

metrics_callback.plot_metrics()

trainer.save_model()
processor.save_pretrained("./segmentation-finetuned")
print("Обучение завершено и модель сохранена!")