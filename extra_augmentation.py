import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np



train_input_dir = "C:/Users/sorvi/Downloads/dataset/train/input"
train_target_dir = "C:/Users/sorvi/Downloads/dataset/train/target"

datasets = ['dataset5']
for dataset in datasets:
    os.makedirs(f"C:/Users/sorvi/Downloads/{dataset}/train/input", exist_ok=True)
    os.makedirs(f"C:/Users/sorvi/Downloads/{dataset}/train/target", exist_ok=True)


image_files = [f for f in os.listdir(train_input_dir) if f.lower().endswith('.png')]


for image_file in image_files:
    image_path_input = os.path.join(train_input_dir, image_file)
    image_path_target = os.path.join(train_target_dir, image_file)
    image_pil_input = Image.open(image_path_input)
    image_pil_target = Image.open(image_path_target)
    rotated_input = image_pil_input.rotate(180)
    rotated_target = image_pil_target.rotate(180)
    rotated_input.save(f"C:/Users/sorvi/Downloads/dataset5/train/input/{image_file}")
    rotated_target.save(f"C:/Users/sorvi/Downloads/dataset5/train/target/{image_file}")