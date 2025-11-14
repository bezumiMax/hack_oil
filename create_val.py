import os
import random
import shutil


input_dir = "C:/Users/sorvi/Downloads/input"
target_dir = "C:/Users/sorvi/Downloads/target"
output_dir = "C:/Users/sorvi/Downloads/dataset"

os.makedirs(f"{output_dir}/train/input", exist_ok=True)
os.makedirs(f"{output_dir}/train/target", exist_ok=True)
os.makedirs(f"{output_dir}/val/input", exist_ok=True)
os.makedirs(f"{output_dir}/val/target", exist_ok=True)

all_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

random.shuffle(all_files)
split_idx = int(0.9 * len(all_files))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

for file in train_files:
    shutil.copy2(f"{input_dir}/{file}", f"{output_dir}/train/input/{file}")
    shutil.copy2(f"{target_dir}/{file}", f"{output_dir}/train/target/{file}")

for file in val_files:
    shutil.copy2(f"{input_dir}/{file}", f"{output_dir}/val/input/{file}")
    shutil.copy2(f"{target_dir}/{file}", f"{output_dir}/val/target/{file}")