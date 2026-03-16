import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_iou(pred_mask, gt_mask, eps=1e-6):
    """Calculates Intersection over Union between predicted boolean masks and ground truth masks."""
    pred_bool = pred_mask > 0.5
    gt_bool = gt_mask > 0.5
    
    # Use negative indexing to sum over the spatial dimensions (H, W) regardless of tensor rank
    intersection = (pred_bool & gt_bool).to(pred_mask.dtype).sum(dim=(-2, -1))
    union = (pred_bool | gt_bool).to(pred_mask.dtype).sum(dim=(-2, -1))
    
    return intersection / (union + eps)

class SAMMultiMaskLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        # Focal loss hyperparameters
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_masks, iou_preds, gt_mask):
        B, num_masks, H, W = pred_masks.shape
        
        # Expand GT mask to match the 3 predicted masks: (B, 3, H, W)
        gt_mask_expanded = gt_mask.expand(-1, num_masks, -1, -1)
        
        # 1. Compute Focal Loss per pixel, then average over spatial dimensions
        # First, get standard BCE with logits (unreduced)
        bce_loss_unreduced = F.binary_cross_entropy_with_logits(pred_masks, gt_mask_expanded, reduction='none')
        
        # Calculate p_t (probability of the true class)
        # Since BCE is -log(p_t), exp(-BCE) gives us p_t directly and stably
        pt = torch.exp(-bce_loss_unreduced)
        
        # Calculate alpha weighting for class imbalance
        alpha_t = gt_mask_expanded * self.alpha + (1 - gt_mask_expanded) * (1 - self.alpha)
        
        # Compute the final focal loss per pixel
        focal_loss_unreduced = alpha_t * (1 - pt) ** self.gamma * bce_loss_unreduced
        
        # Average over spatial dimensions (H, W) -> Shape: (B, 3)
        focal_loss = focal_loss_unreduced.mean(dim=(2, 3)) 
        
        # 2. Compute Dice Loss (working with probabilities, so apply sigmoid)
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_mask_expanded).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_mask_expanded.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5) # Shape: (B, 3)
        
        # Total mask loss per predicted mask (Focal + Dice)
        total_mask_loss = focal_loss + dice_loss # Shape: (B, 3)
        
        # 3. The Minimum-Loss Strategy: Find the index of the mask with the lowest loss
        min_loss, min_loss_idx = torch.min(total_mask_loss, dim=1)  # min_loss shape: (B,)
        
        # 4. Compute actual IoU between predictions and GT to train the IoU head
        with torch.no_grad():
            # Vectorized computation: returns a tensor of shape (B, num_masks) directly
            actual_ious = calc_iou(pred_probs, gt_mask)

        # MSE loss between predicted IoU scores and actual IoU scores
        iou_loss = F.mse_loss(iou_preds, actual_ious)
        
        # Final loss is the batch average of the minimum mask losses + the IoU MSE loss
        return min_loss.mean() + iou_loss