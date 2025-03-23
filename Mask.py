import os
from PIL import Image
import numpy as np

imgs_dir_path = './data/TrainDataset/Imgs'
gt_dir_path = './data/TrainDataset/GT'
mask_dir_path = './Mask'

os.makedirs(mask_dir_path, exist_ok=True)

img_files = [f for f in os.listdir(imgs_dir_path) if os.path.isfile(os.path.join(imgs_dir_path, f))]

for img_filename in img_files:
    img_path = os.path.join(imgs_dir_path, img_filename)
    gt_filename = os.path.splitext(img_filename)[0] + '.png'  
    gt_path = os.path.join(gt_dir_path, gt_filename)

    if os.path.exists(gt_path):
        img = Image.open(img_path).convert("RGBA")
        gt = Image.open(gt_path).convert("L") 

        gt = gt.resize(img.size)

        mask = np.array(gt)
        mask[mask < 255] = 0
        mask = Image.fromarray(mask, 'L')

        masked_img = Image.composite(img, Image.new("RGBA", img.size, 0), mask)

        if masked_img.mode == 'RGBA':
            masked_img = masked_img.convert('RGB')

        mask_img_path = os.path.join(mask_dir_path, os.path.splitext(img_filename)[0] + '.jpg')  # Change file extension to .jpg
        masked_img.save(mask_img_path, 'JPEG')
