import os
import torch
import torch.optim as optim
import numpy as np
import webdataset as wds
from functools import partial
import torchvision.transforms.functional as TF
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

import logging

# --- Import custom modules ---
from dinov3.layers import SelfAttentionBlock
from sam import SAMDINOv3
from loss import SAMMultiMaskLoss 

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

logger = logging.getLogger()

def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ==========================================
# 1. WebDataset Preprocessing Function
# ==========================================
def preprocess_sa1b(sample, image_size=(1024, 1024)):
    """
    Takes a tuple of (PIL_Image, JSON_Dict) from WebDataset, processes them, and returns PyTorch tensors.
    """
    image_pil, data = sample
    
    # 1. Process Image
    image = image_pil.convert("RGB")
    image = TF.resize(image, image_size)
    image = TF.to_tensor(image) # Shape: (3, H, W)

    # 2. Pick a random annotation for this step
    ann = np.random.choice(data['annotations'])
    
    # 3. Process Box Prompt [x, y, w, h] -> [x1, y1, x2, y2]
    x, y, w, h = ann['bbox']
    box_prompt = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
    
    # 4. Process RLE Mask
    rle = ann['segmentation']
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    
    gt_mask_np = mask_utils.decode(rle)
    gt_mask = torch.from_numpy(gt_mask_np).float().unsqueeze(0) 
    gt_mask = TF.resize(gt_mask, image_size, interpolation=TF.InterpolationMode.NEAREST)
    
    return image, box_prompt, gt_mask

# ==========================================
# 2. Main Training Loop
# ==========================================
def train():

    init_logger()

    local_rank = setup_distributed()

    # --- Webdataset setup ---
    # Define your tarball pattern using brace expansion. 
    # Adjust the numbers to match your actual file names.
    tar_url = "/lustre/polis/stf218/scratch/emin/sa1b/sorted/sa_{000000..000999}.tar"
    batch_size = 64 
    
    # resampled=True is the magic wand for multi-node. It causes nodes to randomly sample 
    # tarballs with replacement, meaning you never hit deadlocks at the end of an epoch.
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
        
        .map(preprocess_sa1b, handler=wds.warn_and_continue)
        .batched(batch_size, partial=False)
    )    
    # WebLoader wraps it with multiprocessing workers
    dataloader = wds.WebLoader(
        dataset, 
        batch_size=None,  # Must be None because WebDataset handles batching internally
        num_workers=8, 
        pin_memory=True
    )

    # --- Model Setup ---
    model = SAMDINOv3(model_name='dinov3_vitl16').to(local_rank)
    
    bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported()
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if bf16_ready else torch.float16,
        reduce_dtype=torch.bfloat16 if bf16_ready else torch.float16,
        buffer_dtype=torch.bfloat16 if bf16_ready else torch.float16,
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

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5}, 
        {'params': decoder_params, 'lr': 1e-4}   
    ], weight_decay=0.05)
    
    criterion = SAMMultiMaskLoss().to(local_rank)
    
    # --- Training Loop ---
    # Because resampled=True makes the dataset infinite, we define "epochs" by step count
    steps_per_epoch = 10_000_000 // (batch_size * 16) 
    epochs = 3
    
    # Dataloader is an infinite iterator now
    data_iterator = iter(dataloader)
    
    for epoch in range(epochs):
        model.train()
        
        for step in range(steps_per_epoch):
            # Pull next batch from the stream
            images, boxes, gt_masks = next(data_iterator)
            
            # Note: WebDataset stacks lists of tensors into a final tensor automatically 
            # when you use .batched(). You just need to move them to the GPU.
            # However, because boxes was [x1, y1, x2, y2], WebDataset returns a shape of 
            # (B, 4). Our model expects (B, 1, 4) for a single box per image.
            boxes = boxes.unsqueeze(1) 
            
            images = images.to(local_rank)
            boxes = boxes.to(local_rank)
            gt_masks = gt_masks.to(local_rank)
            
            optimizer.zero_grad()
            pred_masks, iou_preds = model(images, boxes=boxes)

            gt_masks = gt_masks.to(pred_masks.dtype)

            loss = criterion(pred_masks, iou_preds, gt_masks)
            
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                # 1. Detach and clone the local loss so we don't break the autograd graph
                global_loss = loss.detach().clone()
                
                # 2. Average the loss across all 16 GPUs
                # ReduceOp.AVG mathematically sums the tensor across all ranks and divides by world_size
                dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)
                
                # 3. Print the globally averaged loss only on Rank 0
                logger.info(f"Epoch {epoch} | Step {step}/{steps_per_epoch} | Global Loss: {global_loss.item():.4f} ")

    # 1. Force all 16 GPUs to wait for each other before anyone starts saving
    # This prevents timeouts if one node finishes its final batch a few seconds early
    dist.barrier()
    
    if local_rank == 0:
        logger.info("Training complete! Initiating parallel checkpoint save...")

    # 2. Use the modern PyTorch 2.x Distributed Checkpointing API
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
