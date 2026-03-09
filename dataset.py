import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils
import torchvision.transforms.functional as TF

class SA1BDataset(Dataset):
    def __init__(self, image_dir, json_dir, image_size=(1024, 1024)):
        self.image_dir = image_dir
        self.image_size = image_size
        
        # Load all JSON files in the directory
        self.annotations = []
        for file in os.listdir(json_dir):
            if file.endswith('.json'):
                with open(os.path.join(json_dir, file), 'r') as f:
                    data = json.load(f)
                    self.annotations.append(data)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        
        # 1. Load Image
        img_info = data['image']
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Resize image to match model expectations
        image = TF.resize(image, self.image_size)
        image = TF.to_tensor(image) # Shape: (3, H, W), normalized to [0, 1]

        # 2. Pick a random annotation for this training step
        # SA-1B has multiple masks per image; we randomly sample one per forward pass
        ann = np.random.choice(data['annotations'])
        
        # 3. Process Bounding Box Prompt: [x, y, w, h] -> [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        box_prompt = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        
        # 4. Decode Ground Truth Mask from RLE
        rle = ann['segmentation']
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        
        gt_mask_np = mask_utils.decode(rle)
        gt_mask = torch.from_numpy(gt_mask_np).float().unsqueeze(0) # Shape: (1, H_orig, W_orig)
        
        # Resize GT mask to match image size (using nearest exact to preserve binary 0/1)
        gt_mask = TF.resize(gt_mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        
        return image, box_prompt, gt_mask