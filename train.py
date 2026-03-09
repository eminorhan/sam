import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
# Import SelfAttentionBlock from the backbone for FSDP wrapping
from dinov3.layers import SelfAttentionBlock

# Import model definition
from sam import SAMDINOv3 

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def train():
    local_rank = setup_distributed()
    world_size = int(os.environ["WORLD_SIZE"])
    
    # 1. Initialize Dataset and Distributed Sampler
    dataset = SA1BDataset(image_dir="/path/to/sa1b/images", json_dir="/path/to/sa1b/jsons")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=int(os.environ["RANK"]))
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=4, pin_memory=True)
    
    # 2. Initialize Model on CPU first (FSDP handles moving to GPU and sharding)
    model = SAMDINOv3(model_name='dinov3_vitl16', num_masks=3)
    
    # Mixed precision using bfloat16 (highly recommended for H200)
    bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported()
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if bf16_ready else torch.float16,
        reduce_dtype=torch.bfloat16 if bf16_ready else torch.float16,
        buffer_dtype=torch.bfloat16 if bf16_ready else torch.float16,
    )
    
    # ==========================================
    # 1. Activation Checkpointing on blocks
    # ==========================================
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_ac_fn(submodule):
        return isinstance(submodule, SelfAttentionBlock)
        
    apply_activation_checkpointing(
        model, 
        checkpoint_wrapper_fn=non_reentrant_wrapper, 
        check_fn=check_ac_fn
    )

    # ==========================================
    # 2. Compile blocks
    # ==========================================
    # We explicitly iterate through the DINOv3 blocks and compile them individually.
    # This ensures only the heavy transformer blocks are compiled, keeping compilation fast.
    for i, blk in enumerate(model.image_encoder.backbone.blocks):
        model.image_encoder.backbone.blocks[i] = torch.compile(blk)

    # ==========================================
    # 3. FSDP blocks + global model
    # ==========================================
    # Grab the exact memory IDs of our newly wrapped/compiled blocks
    block_ids = {id(blk) for blk in model.image_encoder.backbone.blocks}

    def fsdp_custom_wrap_policy(module, recurse, nonwrapped_numel):
        # If the current module's ID matches one of our compiled blocks, wrap it in its own FSDP unit.
        if id(module) in block_ids:
            return True
        return False

    # FSDP will wrap the returned blocks individually, and then automatically wrap 
    # the rest of the model (Mask Decoder, Prompt Encoder, etc.) into the global root FSDP unit.
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_custom_wrap_policy,
        mixed_precision=mp_policy,
        device_id=local_rank,
        use_orig_params=True, # Critical for torch.compile and separate learning rates
        sync_module_states=True
    )

    # 4. Setup optimizer and loss
    # Separate parameters by checking their names
    backbone_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "image_encoder" in name:
            backbone_params.append(param)
        else:
            # Prompt encoder and mask decoder parameters
            decoder_params.append(param)

    # Apply a 10x smaller learning rate to the backbone
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5}, 
        {'params': decoder_params, 'lr': 1e-4}   
    ])

    criterion = SAMMultiMaskLoss().to(local_rank)
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch) # Important: shuffles data differently each epoch
        
        for batch_idx, (images, boxes, gt_masks) in enumerate(dataloader):
            # Move inputs to local GPU
            images = images.to(local_rank)
            boxes = boxes.to(local_rank)
            gt_masks = gt_masks.to(local_rank)
            
            optimizer.zero_grad()
            
            # Forward pass
            # FSDP automatically gathers the necessary weights for this GPU
            pred_masks, iou_preds = model(images, boxes=boxes)
            
            # Compute loss
            loss = criterion(pred_masks, iou_preds, gt_masks)
            
            # Backward pass and step
            loss.backward()
            optimizer.step()
            
            if local_rank == 0 and batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    if local_rank == 0:
        # Save model (FSDP requires a specific state_dict extraction to stitch shards back together)
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        torch.save(cpu_state, "custom_sam_dinov3_finetuned.pth")
        print("Training complete. Model saved.")

    cleanup_distributed()

if __name__ == "__main__":
    train()