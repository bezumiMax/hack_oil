from PIL import Image, ImageOps
import torch
import numpy as np
import os
from transformers import AutoProcessor
from help_class import SegFormerWithResize

model_path = "C:/Users/sorvi/Downloads/segmentation-finetuned"

processor = AutoProcessor.from_pretrained(model_path)
model = SegFormerWithResize.from_pretrained(model_path)
model.eval()

def padding(image, target_size=3200):
    width, height = image.size
    
    delta_w = target_size - width
    delta_h = target_size - height
    
    padding = (
        delta_w // 2,
        delta_h // 2,  
        delta_w - delta_w // 2,
        delta_h - delta_h // 2
    )
    
    image_padded = ImageOps.expand(image, padding, fill=0)
    return image_padded, padding

def predict_patch(patch):
    inputs = processor(images=patch, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"])
        mask = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()
    
    return mask

def process_large_image(image):
    full_mask = np.zeros((3200, 3200), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            x_start = j * 640
            y_start = i * 640
            x_end = x_start + 640
            y_end = y_start + 640
            
            patch = image.crop((x_start, y_start, x_end, y_end))
            
            mask_patch = predict_patch(patch)
            
            full_mask[y_start:y_end, x_start:x_end] = mask_patch
            
            print(f"Обработан патч ({i},{j})")
    
    return Image.fromarray(full_mask)

predict_dir = "C:/Users/sorvi/Downloads/predict_data/predict_input"
os.makedirs("C:/Users/sorvi/Downloads/answer", exist_ok=True)

image_files = [f for f in os.listdir(predict_dir) if f.lower().endswith('.png')]


for image_file in image_files:
    image_path_input = os.path.join(predict_dir, image_file)
    image_pil_input = Image.open(image_path_input)
    
    image_padded, _ = padding(image_pil_input, 3200)
    
    result_mask = process_large_image(image_padded)

    output_path = os.path.join("C:/Users/sorvi/Downloads/answer", f"{image_file}")
    result_mask.save(output_path)
    print(f"Сохранена маска: {output_path}")

print("Обработка всех изображений завершена!")