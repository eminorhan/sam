import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_iou(pred_mask, gt_mask, eps=1e-6):
    """Calculates Intersection over Union between a predicted boolean mask and a ground truth mask."""
    pred_bool = pred_mask > 0.5
    gt_bool = gt_mask > 0.5
    
    # FIX 2: Use .to(pred_mask.dtype) instead of .float()
    intersection = (pred_bool & gt_bool).to(pred_mask.dtype).sum(dim=(1, 2))
    union = (pred_bool | gt_bool).to(pred_mask.dtype).sum(dim=(1, 2))
    
    return intersection / (union + eps)

class SAMMultiMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_masks, iou_preds, gt_mask):
        B, num_masks, H, W = pred_masks.shape
        
        # Expand GT mask to match the 3 predicted masks: (B, 3, H, W)
        gt_mask_expanded = gt_mask.expand(-1, num_masks, -1, -1)
        
        # 1. Compute BCE Loss per pixel, then average over spatial dimensions
        bce_loss = self.bce(pred_masks, gt_mask_expanded).mean(dim=(2, 3)) # Shape: (B, 3)
        
        # 2. Compute Dice Loss (working with probabilities, so apply sigmoid)
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_mask_expanded).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_mask_expanded.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5) # Shape: (B, 3)
        
        # Total mask loss per predicted mask
        total_mask_loss = bce_loss + dice_loss # Shape: (B, 3)
        
        # 3. The Minimum-Loss Strategy: Find the index of the mask with the lowest loss
        min_loss, min_loss_idx = torch.min(total_mask_loss, dim=1) # min_loss shape: (B,)
        
        # 4. Compute actual IoU between predictions and GT to train the IoU head
        with torch.no_grad():
            actual_ious = torch.stack([calc_iou(pred_probs[:, i], gt_mask.squeeze(1)) for i in range(num_masks)], dim=1)
            
        # MSE loss between predicted IoU scores and actual IoU scores
        iou_loss = F.mse_loss(iou_preds, actual_ious)
        
        # Final loss is the batch average of the minimum mask losses + the IoU MSE loss
        return min_loss.mean() + iou_loss