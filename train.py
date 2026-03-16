import os
import logging
import torch
import torch.optim as optim
import numpy as np
import webdataset as wds
import torchvision.transforms.functional as TF

from functools import partial
from pycocotools import mask as mask_utils

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

# Import custom modules
from dinov3.layers import SelfAttentionBlock
from sam import SAMDINOv3
from loss import SAMMultiMaskLoss 

# Distributed setup and cleanup
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

# Logger
logger = logging.getLogger()

def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Learning rate scheduler
def lr_lambda(current_step, warmup_steps, total_steps):
    """
    Calculates the learning rate multiplier for linear warmup and linear decay.
    """
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, warmup_steps))
    # Linear decay
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

# ==========================================
# 1. Corrected WebDataset Preprocessing 
# ==========================================
def preprocess_sa1b(sample, image_size=(1024, 1024), num_masks_per_image=32):
    """
    Takes a tuple of (PIL_Image, JSON_Dict) from WebDataset.
    Properly scales SA-1B absolute coordinates to match the resized image.
    """
    image_pil, data = sample
    
    # 1. Calculate scaling factors before resizing
    orig_w, orig_h = image_pil.size
    scale_w = image_size[1] / orig_w  # image_size is (H, W)
    scale_h = image_size[0] / orig_h
    
    # 2. Process Image
    image = image_pil.convert("RGB")
    image = TF.resize(image, image_size)
    image = TF.to_tensor(image) # Shape: (3, H, W)

    # 3. Pick multiple annotations (pad with replacement if fewer than num_masks)
    anns = data['annotations']
    if len(anns) >= num_masks_per_image:
        selected_anns = np.random.choice(anns, num_masks_per_image, replace=False)
    else:
        selected_anns = np.random.choice(anns, num_masks_per_image, replace=True)
    
    boxes, pt_coords, pt_labels, gt_masks = [], [], [], []
    
    for ann in selected_anns:
        # Box Prompt: Convert XYWH to XYXY AND apply scaling
        x, y, w, h = ann['bbox']
        boxes.append([
            x * scale_w, 
            y * scale_h, 
            (x + w) * scale_w, 
            (y + h) * scale_h
        ])
        
        # Point Prompt: SA-1B format is [[x, y]]. Apply scaling.
        pt = ann['point_coords'][0]
        pt_coords.append([[pt[0] * scale_w, pt[1] * scale_h]]) # Shape (1, 2)
        pt_labels.append([1])                                  # Shape (1)
        
        # RLE Mask
        rle = ann['segmentation']
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        
        gt_mask_np = mask_utils.decode(rle)
        gt_mask = torch.from_numpy(gt_mask_np).float().unsqueeze(0) # (1, H, W)
        
        # Resize mask to match image
        gt_mask = TF.resize(gt_mask, image_size, interpolation=TF.InterpolationMode.NEAREST)
        gt_masks.append(gt_mask)

    # Stack into tensors
    box_prompts = torch.tensor(boxes, dtype=torch.float32)         # (M, 4)
    point_coords = torch.tensor(pt_coords, dtype=torch.float32)    # (M, 1, 2)
    point_labels = torch.tensor(pt_labels, dtype=torch.long)       # (M, 1)
    gt_masks = torch.stack(gt_masks, dim=0)                        # (M, 1, H, W)
    
    return image, box_prompts, point_coords, point_labels, gt_masks

# ==========================================
# 2. Main Training Loop
# ==========================================
def train():

    init_logger()

    local_rank = setup_distributed()

    # --- Webdataset setup ---
    # Define tarball pattern using brace expansion. 
    tar_url = "/lustre/polis/stf218/scratch/emin/sa1b/sorted/sa_{000000..000999}.tar"
    model_name = 'dinov3_vit7b16'
    batch_size = 16
    num_masks_per_image = 32
    image_size = (1024, 1024)
    log_steps = 100
    grad_norm_clip = 1.0
    
    # resampled=True causes nodes to randomly sample tarballs with replacement
    dataset = (
        wds.WebDataset(
            tar_url, 
            resampled=True, 
            nodesplitter=wds.split_by_node,
        )
        .shuffle(1000, initial=1000)
        
        # handler=wds.warn_and_continue tells the pipeline to drop corrupted files 
        # and move to the next one instead of crashing the script.
        .decode("pil", handler=wds.warn_and_continue) 
        
        # "jpg;jpeg" tells it to accept either extension to match with the json
        .to_tuple("jpg;jpeg", "json", handler=wds.warn_and_continue) 
        
        # pass arguments to preprocessing function via partial
        .map(partial(preprocess_sa1b, image_size=image_size, num_masks_per_image=num_masks_per_image), handler=wds.warn_and_continue)
        .batched(batch_size, partial=False)
    )

    # WebLoader wraps it with multiprocessing workers
    dataloader = wds.WebLoader(
        dataset, 
        batch_size=None,  # must be None because WebDataset handles batching internally
        num_workers=8, 
        pin_memory=True
    )

    # --- Model Setup ---
    model = SAMDINOv3(model_name=model_name).to(local_rank)
    
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # --- 1. Activation Checkpointing ---
    non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        model, 
        checkpoint_wrapper_fn=non_reentrant_wrapper, 
        check_fn=lambda submodule: isinstance(submodule, SelfAttentionBlock)
    )

    # --- 2. Compile Blocks ---
    for i, blk in enumerate(model.image_encoder.backbone.blocks):
        model.image_encoder.backbone.blocks[i] = torch.compile(blk)

    # --- 3. FSDP Wrapping ---
    block_ids = {id(blk) for blk in model.image_encoder.backbone.blocks}
    def fsdp_custom_wrap_policy(module, recurse, nonwrapped_numel):
        return id(module) in block_ids

    model = FSDP(
        model,
        auto_wrap_policy=fsdp_custom_wrap_policy,
        mixed_precision=mp_policy,
        device_id=local_rank,
        use_orig_params=True, 
        sync_module_states=True
    )
    
    # --- Optimizer with Split Learning Rates ---
    backbone_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_encoder" in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    # Using different learning rates for backbone vs. decoder parameters
    optimizer = optim.AdamW([{'params': backbone_params, 'lr': 3e-5}, {'params': decoder_params, 'lr': 3e-4}], weight_decay=0.05)
    
    criterion = SAMMultiMaskLoss().to(local_rank)
    
    # --- Training Loop ---
    # Because resampled=True makes the dataset infinite, we define "epochs" by step count
    steps_per_epoch = 10_000_000 // (batch_size * 16) 
    epochs = 3

    # --- LR Scheduler ---
    total_steps = epochs * steps_per_epoch
    warmup_steps = 100

    # Bind the fixed arguments using partial
    bound_lr_lambda = partial(lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, bound_lr_lambda)

    # Dataloader is an infinite iterator now
    data_iterator = iter(dataloader)
    
    for epoch in range(epochs):
        model.train()
        
        for step in range(steps_per_epoch):

            # Yield inputs
            images, boxes, pt_coords, pt_labels, gt_masks = next(data_iterator)
            
            # boxes: (B, M, 4) | pt_coords: (B, M, 1, 2) | gt_masks: (B, M, 1, H, W)
            boxes = boxes.to(local_rank)   
            pt_coords = pt_coords.to(local_rank)   
            pt_labels = pt_labels.to(local_rank)
            
            images = images.to(local_rank)
            gt_masks = gt_masks.to(local_rank)
            
            optimizer.zero_grad()
            
            prompt_choice = np.random.choice(["box_only", "point_only", "both"])
            
            if prompt_choice == "box_only":
                model_points = None
                model_boxes = boxes
            elif prompt_choice == "point_only":
                model_points = (pt_coords, pt_labels)
                model_boxes = None
            else:
                model_points = (pt_coords, pt_labels)
                model_boxes = boxes
            
            # Forward pass outputs shapes: (B, M, 3, H, W) and (B, M, 3)
            pred_masks, iou_preds = model(images, points=model_points, boxes=model_boxes)

            gt_masks = gt_masks.to(pred_masks.dtype)

            # Flatten the B and M dimensions so the loss function processes them like a standard batch
            # Shape becomes: (B*M, 3, H, W) vs GT of (B*M, 1, H, W)
            loss = criterion(
                pred_masks.flatten(0, 1), 
                iou_preds.flatten(0, 1), 
                gt_masks.flatten(0, 1)
            )

            loss.backward()

            # FSDP-safe gradient clipping: this safely handles cross-GPU communication to clip sharded gradients
            model.clip_grad_norm_(grad_norm_clip)

            optimizer.step()
            scheduler.step()

            if step % log_steps == 0:
                # Detach and clone the local loss so we don't break the autograd graph
                global_loss = loss.detach().clone()
                
                # Average the loss across all ranks
                dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)
                
                # Fetch current learning rates for logging
                lr_backbone = optimizer.param_groups[0]['lr']
                lr_decoder = optimizer.param_groups[1]['lr']

                # Print the globally averaged loss only on Rank 0
                logger.info(f"epoch {epoch} | step {step}/{steps_per_epoch} | loss: {global_loss.item():.4f} | lr (backbone): {lr_backbone:.2e} | lr (decoder): {lr_decoder:.2e} ")

    # Force all ranks to wait for each other before anyone starts saving
    # This prevents timeouts if one node finishes its final batch a few seconds early
    dist.barrier()
    
    if local_rank == 0:
        logger.info("Training complete! Initiating parallel checkpoint save...")

    # Use Distributed Checkpointing API
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
    
    # Define a folder to hold the shards instead of a single file
    checkpoint_dir = "/lustre/polis/stf218/scratch/emin/sam_dinov3_checkpoint"
    
    # Get the sharded state dict (no network transfer required!)
    state_dict = get_state_dict(model, options=StateDictOptions())
    
    # Save shards in parallel directly to Lustre
    dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_dir)

    if local_rank == 0:
        logger.info(f"✅ Checkpoint successfully saved to {checkpoint_dir}")

if __name__ == "__main__":
    train()
    cleanup_distributed()
