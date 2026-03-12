import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils
import torchvision.transforms.functional as TF

class SA1BDataset(Dataset):
    def __init__(self, image_base_dir, json_base_dir, image_size=(1024, 1024)):
        self.image_base_dir = Path(image_base_dir)
        self.json_base_dir = Path(json_base_dir)
        self.image_size = image_size
        
        # We will only store string paths to save CPU RAM
        self.samples = []
        
        print("Indexing SA-1B dataset shards...")
        
        # Iterate through all shard folders (e.g., sa_000001, sa_000002)
        for shard_dir in self.json_base_dir.iterdir():
            if not shard_dir.is_dir():
                continue
                
            shard_name = shard_dir.name # e.g., 'sa_000001'
            image_shard_dir = self.image_base_dir / shard_name
            
            # Skip if the corresponding image shard folder doesn't exist
            if not image_shard_dir.exists():
                print(f"Warning: Image shard {image_shard_dir} missing. Skipping.")
                continue
                
            # Find all JSONs in this annotation shard
            for json_path in shard_dir.glob("*.json"):
                # SA-1B images have the exact same base name as the JSON but end in .jpg
                image_name = json_path.stem + ".jpg"
                image_path = image_shard_dir / image_name
                
                # Only append if both files exist
                if image_path.exists():
                    self.samples.append((str(image_path), str(json_path)))
                    
        print(f"Successfully indexed {len(self.samples)} image-annotation pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        
        # 1. Lazy-load the JSON data only when requested by the dataloader worker
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 2. Load and resize the Image
        image = Image.open(img_path).convert("RGB")
        image = TF.resize(image, self.image_size)
        image = TF.to_tensor(image) # Shape: (3, H, W)

        # 3. Pick a random annotation (mask) for this specific training step
        ann = np.random.choice(data['annotations'])
        
        # 4. Process Bounding Box Prompt: [x, y, w, h] -> [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        box_prompt = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        
        # 5. Decode Ground Truth Mask from RLE to binary tensor
        rle = ann['segmentation']
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        
        gt_mask_np = mask_utils.decode(rle)
        gt_mask = torch.from_numpy(gt_mask_np).float().unsqueeze(0) 
        
        # Resize GT mask to match image size (must use NEAREST to preserve binary 0/1 values)
        gt_mask = TF.resize(gt_mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        
        return image, box_prompt, gt_mask