import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random



folder1_dir = ""
folder2_dir = ""
folder3_dir = ""

image_files1 = [f for f in os.listdir(folder1_dir) if f.lower().endswith('.png')]
image_files2 = [f for f in os.listdir(folder2_dir) if f.lower().endswith('.png')]
image_files3 = [f for f in os.listdir(folder3_dir) if f.lower().endswith('.png')]



def pixel_voting(mask1, mask2, mask3):
    height, width = mask1.shape
    final_mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            votes = [mask1[i, j], mask2[i, j], mask3[i, j]]
            if votes[0] == votes[1] or votes[0] == votes[2]:
                final_mask[i, j] = votes[0]
            elif votes[1] == votes[2]:
                final_mask[i, j] = votes[1]
            else:
                final_mask[i, j] = random.choice(votes)
    
    return final_mask

for image_file1, image_file2, image_file3 in zip(image_files1, image_files2, image_files3):
    image_path1 = os.path.join(folder1_dir, image_file1)
    image_path2 = os.path.join(folder2_dir, image_file2)
    image_path3 = os.path.join(folder3_dir, image_file3)
    image1 = Image.open(image_path1).convert('L')
    image2 = Image.open(image_path2).convert('L')
    image3 = Image.open(image_path3).convert('L')
    mask1 = np.array(image1)
    mask2 = np.array(image2)
    mask3 = np.array(image3)
    final_mask = pixel_voting(mask1, mask2, mask3)
    result_image = Image.fromarray(final_mask)
    dir = "path/to/folder/"
    result_image.save(f"{dir}{image_file1}")