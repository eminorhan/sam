import torch
import torch.nn as nn
import torch.nn.functional as F

# change the following vars according to your setup
TORCH_HUB_PATH = "/lustre/blizzard/stf218/scratch/emin/torch_hub"  # this is where the dinov3 pth checkpoints are stored
DINOV3_REPO_PATH = "/lustre/blizzard/stf218/scratch/emin/dinov3"  # dinov3 repo path

torch.hub.set_dir(TORCH_HUB_PATH)

class DINOv3ImageEncoder(nn.Module):
    """
    Loads a DINOv3 backbone, extracts patch features, and projects 
    them to the required embedding dimension for the SAM decoder.
    """
    def __init__(self, model_name='dinov3_vitl16', out_dim=256):
        super().__init__()
        # Load pre-trained DINOv3 from torch.hub
        self.backbone = torch.hub.load(DINOV3_REPO_PATH, "dinov3_vitl16", source="local", weights=f"{TORCH_HUB_PATH}/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", in_chans=3)
                    
        in_dim = self.backbone.embed_dim 
        
        # SAM uses a neck to project the ViT dimensionality to out_dim (256)
        self.neck = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # DINOv3 forward_features returns a dict. We need the normalized patch tokens.
        ret = self.backbone.forward_features(x)
        patch_tokens = ret['x_norm_patchtokens'] # Shape: (B, N, D)
        
        # Assuming square inputs and DINOv3's patch size of 16
        h_feat, w_feat = H // 16, W // 16
        
        # Reshape the sequence of tokens back into a 2D spatial grid
        features = patch_tokens.transpose(1, 2).reshape(B, -1, h_feat, w_feat)
        return self.neck(features)

class PromptEncoder(nn.Module):
    """
    Encodes points and bounding boxes into sparse prompt embeddings.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Fixed gaussian matrix for positional encoding
        self.register_buffer("pe_matrix", torch.randn((2, embed_dim // 2)))
        
        # Learned embeddings: 0=Background, 1=Foreground, 2=Top-Left Box, 3=Bottom-Right Box
        self.point_embeddings = nn.Embedding(4, embed_dim)
        
    def _pe_encoding(self, coords):
        # Maps coordinates from [0, 1] to the positional encoding space
        coords = 2 * coords - 1
        coords = coords @ self.pe_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        
    def forward(self, points=None, boxes=None, image_size=(1024, 1024)):
        B = points[0].shape[0] if points is not None else (boxes.shape[0] if boxes is not None else 1)
        sparse_embeddings = torch.empty((B, 0, self.embed_dim), device=self.point_embeddings.weight.device)
        
        # Process Points
        if points is not None:
            coords, labels = points
            coords = coords.float()
            coords[:, :, 0] /= image_size[1]
            coords[:, :, 1] /= image_size[0]
            
            pe = self._pe_encoding(coords)
            pt_emb = pe + self.point_embeddings(labels)
            sparse_embeddings = torch.cat([sparse_embeddings, pt_emb], dim=1)
            
        # Process Boxes
        if boxes is not None:
            coords = boxes.reshape(B, -1, 2, 2).float() # (B, N, 2(points), 2(x,y))
            coords[:, :, :, 0] /= image_size[1]
            coords[:, :, :, 1] /= image_size[0]
            
            pe = self._pe_encoding(coords)
            
            tl_embed = pe[:, :, 0, :] + self.point_embeddings(torch.tensor(2, device=pe.device))
            br_embed = pe[:, :, 1, :] + self.point_embeddings(torch.tensor(3, device=pe.device))
            sparse_embeddings = torch.cat([sparse_embeddings, tl_embed, br_embed], dim=1)
            
        return sparse_embeddings

class MultiMaskDecoder(nn.Module):
    def __init__(self, transformer_dim=256, num_heads=8, num_masks=3):
        super().__init__()
        self.num_masks = num_masks
        
        self.self_attn = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        self.cross_attn_token_to_image = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        self.cross_attn_image_to_token = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 4),
            nn.ReLU(),
            nn.Linear(transformer_dim * 4, transformer_dim)
        )
        self.norms = nn.ModuleList([nn.LayerNorm(transformer_dim) for _ in range(4)])
        
        # UPDATE: 1 token for IoU prediction, plus 'num_masks' tokens for the actual masks
        self.num_mask_tokens = num_masks + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # UPDATE: A list of MLPs to generate hypernetwork weights for EACH mask
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.ReLU(),
                nn.Linear(transformer_dim, transformer_dim // 8)
            ) for _ in range(self.num_masks)
        ])
        
        # UPDATE: An MLP head to predict the IoU score for each of the 3 masks
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, self.num_masks)
        )
        
    def forward(self, image_embeddings, prompt_embeddings):
        B, C, H, W = image_embeddings.shape
        img_feats = image_embeddings.flatten(2).transpose(1, 2)
        
        # Prepend the IoU token and the 3 mask tokens to the prompt sequence
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([mask_tokens, prompt_embeddings], dim=1) 
        
        # Transformer computations remain the same
        tokens2, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norms[0](tokens + tokens2)
        
        tokens2, _ = self.cross_attn_token_to_image(tokens, img_feats, img_feats)
        tokens = self.norms[1](tokens + tokens2)
        
        tokens = self.norms[2](tokens + self.mlp(tokens))
        
        img_feats2, _ = self.cross_attn_image_to_token(img_feats, tokens, tokens)
        img_feats = self.norms[3](img_feats + img_feats2)
        
        upscaled_embedding = self.output_upscaling(img_feats.transpose(1, 2).reshape(B, C, H, W)) 
        
        # UPDATE: Split tokens. Token 0 is IoU, Tokens 1 to 3 are masks.
        iou_token_out = tokens[:, 0, :]
        mask_tokens_out = tokens[:, 1: (1 + self.num_masks), :] # Shape: (B, 3, C)
        
        # Predict 3 masks
        b, c, h, w = upscaled_embedding.shape
        masks = []
        for i in range(self.num_masks):
            hyper_in = self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            mask_i = (upscaled_embedding * hyper_in.view(B, c, 1, 1)).sum(dim=1, keepdim=True)
            masks.append(mask_i)
            
        masks = torch.cat(masks, dim=1) # Shape: (B, 3, H, W)
        
        # Predict IoU scores using the dedicated IoU token
        iou_pred = self.iou_prediction_head(iou_token_out) # Shape: (B, 3)
        
        return masks, iou_pred

# The wrapper model is updated to return both outputs
class SAMDINOv3(nn.Module):
    def __init__(self, model_name='dinov3_vitb16', num_masks=3):
        super().__init__()
        self.num_masks = num_masks
        self.image_encoder = DINOv3ImageEncoder(model_name) # From previous snippet
        self.prompt_encoder = PromptEncoder(embed_dim=256)  # From previous snippet
        self.mask_decoder = MultiMaskDecoder(transformer_dim=256, num_masks=num_masks)
        
    def forward(self, image, points=None, boxes=None):
        image_size = (image.shape[2], image.shape[3])
        image_embeddings = self.image_encoder(image)
        sparse_embeddings = self.prompt_encoder(points, boxes, image_size)
        
        masks, iou_preds = self.mask_decoder(image_embeddings, sparse_embeddings)
        masks = F.interpolate(masks, size=image_size, mode="bilinear", align_corners=False)
        return masks, iou_preds


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize custom model
    model = SAMDINOv3(model_name='dinov3_vitl16').to(device)
    model.eval()
    print(f"Model: {model}")

    # 2. Dummy Inputs
    B, C, H, W = 2, 3, 1024, 1024
    dummy_image = torch.randn(B, C, H, W).to(device)

    # --- Point Prompt Setup ---
    # N points per batch: coords shape (B, N, 2), labels shape (B, N)
    # Label '1' = Foreground, Label '0' = Background
    point_coords = torch.tensor([[[500, 500], [200, 200]], [[600, 400], [100, 100]]]).to(device)
    point_labels = torch.tensor([[1, 0], [1, 1]]).to(device)
    points = (point_coords, point_labels)

    # --- Box Prompt Setup ---
    # N boxes per batch: shape (B, N, 4) in format [x1, y1, x2, y2]
    boxes = torch.tensor([[[100, 100, 400, 400]], [[300, 300, 800, 800]]]).to(device)

    # 3. Forward Pass
    with torch.no_grad():
        # You can prompt with points, boxes, or both simultaneously
        pred_masks, iou_preds = model(dummy_image, points=points, boxes=boxes)

    # Outputs logits of shape (B, 1, H, W) -> Pass through sigmoid for probabilities
    mask_probs = torch.sigmoid(pred_masks)
    print(f"Output Mask Shape: {mask_probs.shape}")
