import os
import glob
import logging
import argparse
import tomllib
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

# Import DCP APIs
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions

# Import custom modules
from dinov3.layers import SelfAttentionBlock
from sam import SAMDINOv3
from loss import SAMMultiMaskLoss 

# Distributed setup and cleanup
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

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
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize to use dinov3 backbones 

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

def verify_training_health(model, optimizer):
    """Finds the first locally owned 2D weight matrix to verify gradients/momentum."""
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1. Ensure this GPU owns the shard
        # 2. Look for 'weight' in the name (usually matrices, not tokens/biases)
        if param.numel() > 0 and 'weight' in name and 'norm' not in name:
            
            state = optimizer.state.get(param)
            if state and 'exp_avg' in state:
                mom_norm = state['exp_avg'].norm().item()
                
                # If we found a healthy, non-zero momentum, print it and stop!
                if mom_norm > 0.0:
                    weight_norm = param.norm().item()
                    logger.info(f"✅ HEALTHY LAYER FOUND | Param: {name}")
                    logger.info(f"   Weight Norm: {weight_norm:.4f} | Momentum Norm: {mom_norm:.4f}")
                    return # We are good!
                
    logger.warning("⚠️ WARNING: Could not find ANY weight matrix with non-zero momentum on this rank!")

# ==========================================
# 2. Main Training Loop
# ==========================================
def train(config):

    init_logger()

    local_rank, global_rank, world_size = setup_distributed()

    # --- Read config ---
    data_path = config['dataset']['data_path']
    model_name = config['model']['model_name']
    run_name = config['training']['run_name']
    batch_size = config['training']['batch_size']
    num_masks_per_image = config['training']['num_masks_per_image']
    image_size = tuple(config['training']['image_size'])  # convert list to tuple
    log_steps = config['training']['log_steps']
    ckpt_steps = config['training']['ckpt_steps']
    grad_norm_clip = config['training']['grad_norm_clip']
    epochs = config['training']['epochs']
    warmup_steps = config['training']['warmup_steps']
    backbone_lr = config['optimizer']['backbone_lr']
    decoder_lr = config['optimizer']['decoder_lr']

    # --- Base Checkpoint Directory ---
    checkpoint_base_dir = os.path.join("outputs", run_name, "checkpoint")
    if global_rank == 0:
        os.makedirs(checkpoint_base_dir, exist_ok=True)
    dist.barrier()

    # --- Webdataset setup ---
    # resampled=True causes nodes to randomly sample tarballs with replacement
    dataset = (
        wds.WebDataset(
            data_path, 
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
        num_workers=4, 
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
    optimizer = optim.AdamW([{'params': backbone_params, 'lr': backbone_lr}, {'params': decoder_params, 'lr': decoder_lr}], weight_decay=0.05)
    criterion = SAMMultiMaskLoss().to(local_rank)
    
    # --- Training Loop ---
    # Because resampled=True makes the dataset infinite, we define "epochs" by step count
    steps_per_epoch = 11_000_000 // (batch_size * world_size) 
    total_steps = epochs * steps_per_epoch

    # --- LR scheduler --
    bound_lr_lambda = partial(lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, bound_lr_lambda)

    # --- Resume from checkpoint ---
    start_step = 0
    
    # 1. Find the latest checkpoint folder
    latest_ckpt_path = None
    step_dirs = glob.glob(os.path.join(checkpoint_base_dir, "step_*"))
    if step_dirs:
        step_nums = [int(os.path.basename(d).split("_")[1]) for d in step_dirs]
        latest_step = max(step_nums)
        latest_ckpt_path = os.path.join(checkpoint_base_dir, f"step_{latest_step}")

    # 2. Load states if checkpoint exists
    if latest_ckpt_path is not None:
        logger.info(f"Resuming training from checkpoint: {latest_ckpt_path}")
            
        model_state_dict, opt_state_dict = get_state_dict(model, optimizer, options=StateDictOptions())
        
        checkpoint_data = {
            "model": model_state_dict,
            "optimizer": opt_state_dict,
            "scheduler": scheduler.state_dict(), 
            "step": 0
        }
        
        dcp.load(
            state_dict=checkpoint_data, 
            checkpoint_id=latest_ckpt_path,
            planner=DefaultLoadPlanner(allow_partial_load=True)            
        )
        
        set_state_dict(
            model, 
            optimizer, 
            model_state_dict=checkpoint_data["model"], 
            optim_state_dict=checkpoint_data["optimizer"],
            options=StateDictOptions(strict=False)            
        )
        scheduler.load_state_dict(checkpoint_data["scheduler"])

        # # 🟢 CHECK 2: Verify the loaded state applied correctly
        # verify_training_health(model, optimizer)

        start_step = checkpoint_data["step"] + 1

    # Dataloader is an infinite iterator
    data_iterator = iter(dataloader)
    model.train()
    
    logger.info(f"Starting training loop from step {start_step} to {total_steps}...")

    for step in range(start_step, total_steps):
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

        # Logging
        if step % log_steps == 0:
            # Detach and clone the local loss so we don't break the autograd graph
            global_loss = loss.detach().clone()
            
            # Average the loss across all ranks
            dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)
            
            # Fetch current learning rates for logging
            lr_backbone = optimizer.param_groups[0]['lr']
            lr_decoder = optimizer.param_groups[1]['lr']

            # Print the globally averaged loss only on Rank 0
            logger.info(f"step {step}/{total_steps} | loss: {global_loss.item():.4f} | lr (backbone): {lr_backbone:.2e} | lr (decoder): {lr_decoder:.2e} ")

        # Checkpointing
        if step > 0 and step % ckpt_steps == 0:
            dist.barrier()

            # # 🟢 CHECK 1: Verify the live state before saving
            # verify_training_health(model, optimizer) # comment out after checking

            ckpt_path = os.path.join(checkpoint_base_dir, f"step_{step}")
            logger.info(f"Saving checkpoint at step {step} to {ckpt_path}...")

            model_state_dict, opt_state_dict = get_state_dict(model, optimizer, options=StateDictOptions())
            
            save_data = {
                "model": model_state_dict,
                "optimizer": opt_state_dict,
                "scheduler": scheduler.state_dict(),
                "step": step
            }
            
            dcp.save(state_dict=save_data, checkpoint_id=ckpt_path)
            logger.info("✅ Checkpoint successfully saved.")

    # Force all ranks to wait for each other before anyone starts saving
    # This prevents timeouts if one node finishes its final batch a few seconds early
    dist.barrier()
    
    # Final save at the end of training
    logger.info("Training complete! Initiating final checkpoint save...")
    final_ckpt_path = os.path.join(checkpoint_base_dir, "final")
    model_state_dict, opt_state_dict = get_state_dict(model, optimizer, options=StateDictOptions())
    
    save_data = {
        "model": model_state_dict,
        "optimizer": opt_state_dict,
        "scheduler": scheduler.state_dict(),
        "step": total_steps
    }
    
    dcp.save(state_dict=save_data, checkpoint_id=final_ckpt_path)
    logger.info(f"✅ Final checkpoint successfully saved to {final_ckpt_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distributed training of segment-anything models with dinov3 image encoder")
    parser.add_argument("--config", type=str, default="configs/train_config.toml", help="Path to the toml configuration file")
    args = parser.parse_args()

    # Load the toml file
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    train(config)
    cleanup_distributed()
