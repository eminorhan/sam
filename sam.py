import torch
import torch.nn as nn
import torch.nn.functional as F

# change the following vars according to your setup
TORCH_HUB_PATH = "/lustre/polis/stf218/scratch/emin/torch_hub"  # this is where the dinov3 pth checkpoints are stored
DINOV3_REPO_PATH = "/lustre/polis/stf218/scratch/emin/dinov3"  # dinov3 repo path

torch.hub.set_dir(TORCH_HUB_PATH)

BACKBONE_CKPT_DICT = {
    "dinov3_vit7b16_3D": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "dinov3_vith16plus_3D": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vitl16_3D": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitb16_3D": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}

class DINOv3ImageEncoder(nn.Module):
    """
    Loads a DINOv3 backbone, extracts patch features, and projects 
    them to the required embedding dimension for the SAM decoder.
    """
    def __init__(self, model_name, out_dim=256):
        super().__init__()

        self.backbone = torch.hub.load(
            DINOV3_REPO_PATH, 
            model_name, 
            source="local", 
            weights=f"{TORCH_HUB_PATH}/checkpoints/{BACKBONE_CKPT_DICT[model_name]}",
            pretrained=True, 
            use_fa3=True
        )
                    
        in_dim = self.backbone.embed_dim 
        
        self.neck = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ret = self.backbone.forward_features(x)
        patch_tokens = ret['x_norm_patchtokens'] # Shape: (B, N, D)
        
        h_feat, w_feat = H // 16, W // 16
        features = patch_tokens.transpose(1, 2).reshape(B, -1, h_feat, w_feat)
        return self.neck(features)

class PromptEncoder(nn.Module):
    """
    Encodes points and bounding boxes into sparse prompt embeddings.
    Also provides dense positional encodings for the image features.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.register_buffer("pe_matrix", torch.randn((2, embed_dim // 2)))
        # Learned embeddings: 0=Bg, 1=Fg, 2=Top-Left Box, 3=Bottom-Right Box
        self.point_embeddings = nn.Embedding(4, embed_dim)
        
    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords.to(self.pe_matrix.dtype) 
        coords = coords @ self.pe_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def get_dense_pe(self, image_size):
        # FIX: Added dense PE generation for the image features (H/16, W/16)
        h, w = image_size[0] // 16, image_size[1] // 16
        grid = torch.ones((h, w), device=self.pe_matrix.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        coords = torch.stack([x_embed, y_embed], dim=-1)
        
        target_dtype = self.point_embeddings.weight.dtype
        dense_pe = self._pe_encoding(coords).to(target_dtype)
        # Output shape: (1, 256, H, W)
        return dense_pe.permute(2, 0, 1).unsqueeze(0)

    def forward(self, points=None, boxes=None, image_size=(1024, 1024)):
        B = points[0].shape[0] if points is not None else (boxes.shape[0] if boxes is not None else 1)
        target_dtype = self.point_embeddings.weight.dtype
        
        sparse_embeddings = torch.empty(
            (B, 0, self.embed_dim), 
            device=self.point_embeddings.weight.device,
            dtype=target_dtype
        )
        
        # FIX: Out-of-place modification & explicit target_dtype casting
        if points is not None:
            coords, labels = points
            coords_norm = coords.to(target_dtype).clone()
            coords_norm[..., 0] = coords_norm[..., 0] / image_size[1]
            coords_norm[..., 1] = coords_norm[..., 1] / image_size[0]
            
            pe = self._pe_encoding(coords_norm).to(target_dtype)
            pt_emb = pe + self.point_embeddings(labels)
            sparse_embeddings = torch.cat([sparse_embeddings, pt_emb], dim=1)
            
        if boxes is not None:
            coords = boxes.reshape(B, -1, 2, 2).to(target_dtype).clone()
            coords[..., 0] = coords[..., 0] / image_size[1]
            coords[..., 1] = coords[..., 1] / image_size[0]
            
            pe = self._pe_encoding(coords).to(target_dtype)
            tl_embed = pe[:, :, 0, :] + self.point_embeddings(torch.tensor(2, device=pe.device))
            br_embed = pe[:, :, 1, :] + self.point_embeddings(torch.tensor(3, device=pe.device))
            sparse_embeddings = torch.cat([sparse_embeddings, tl_embed, br_embed], dim=1)
            
        return sparse_embeddings

# FIX: Extracted Attention sequence into a proper Decoder Block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, transformer_dim=256, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        self.cross_attn_token_to_image = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        self.cross_attn_image_to_token = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 4),
            nn.ReLU(),
            nn.Linear(transformer_dim * 4, transformer_dim)
        )
        self.norms = nn.ModuleList([nn.LayerNorm(transformer_dim) for _ in range(4)])

    def forward(self, tokens, img_feats, orig_tokens, image_pe):
        # 1. Self attention (add orig_tokens to queries and keys)
        q = k = tokens + orig_tokens
        tokens2, _ = self.self_attn(q, k, tokens)
        tokens = self.norms[0](tokens + tokens2)
        
        # 2. Token to Image cross-attention (add PE to image, orig_tokens to tokens)
        q = tokens + orig_tokens
        k = img_feats + image_pe
        tokens2, _ = self.cross_attn_token_to_image(q, k, img_feats)
        tokens = self.norms[1](tokens + tokens2)
        
        # 3. MLP
        tokens = self.norms[2](tokens + self.mlp(tokens))
        
        # 4. Image to Token cross-attention
        q = img_feats + image_pe
        k = tokens + orig_tokens
        img_feats2, _ = self.cross_attn_image_to_token(q, k, tokens)
        img_feats = self.norms[3](img_feats + img_feats2)
        
        return tokens, img_feats

class MultiMaskDecoder(nn.Module):
    def __init__(self, transformer_dim=256, num_heads=8, num_masks=3, num_layers=2):
        super().__init__()
        self.num_masks = num_masks
        
        # FIX: Instantiate exactly 2 decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(transformer_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.num_mask_tokens = num_masks + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.ReLU(),
                nn.Linear(transformer_dim, transformer_dim // 8)
            ) for _ in range(self.num_masks)
        ])
        
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, self.num_masks)
        )
        
    def forward(self, image_embeddings, prompt_embeddings, image_pe):
        B, C, H, W = image_embeddings.shape
        img_feats = image_embeddings.flatten(2).transpose(1, 2)
        image_pe = image_pe.flatten(2).transpose(1, 2).expand(B, -1, -1)
        
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([mask_tokens, prompt_embeddings], dim=1) 
        orig_tokens = tokens.clone() # Keep orig tokens for re-addition in attention
        
        # FIX: Pass through 2 layers, providing orig_tokens and image_pe
        for layer in self.layers:
            tokens, img_feats = layer(tokens, img_feats, orig_tokens, image_pe)
        
        upscaled_embedding = self.output_upscaling(img_feats.transpose(1, 2).reshape(B, C, H, W)) 
        
        iou_token_out = tokens[:, 0, :]
        mask_tokens_out = tokens[:, 1: (1 + self.num_masks), :] 
        
        b, c, h, w = upscaled_embedding.shape
        masks = []
        for i in range(self.num_masks):
            hyper_in = self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            mask_i = (upscaled_embedding * hyper_in.view(B, c, 1, 1)).sum(dim=1, keepdim=True)
            masks.append(mask_i)
            
        masks = torch.cat(masks, dim=1) # Shape: (B, 3, H, W)
        iou_pred = self.iou_prediction_head(iou_token_out) # Shape: (B, 3)
        
        return masks, iou_pred

class SAMDINOv3(nn.Module):
    def __init__(self, model_name, num_masks=3):
        super().__init__()
        self.num_masks = num_masks
        self.image_encoder = DINOv3ImageEncoder(model_name) 
        self.prompt_encoder = PromptEncoder(embed_dim=256) 
        self.mask_decoder = MultiMaskDecoder(transformer_dim=256, num_masks=num_masks)
        
    def forward(self, image, points=None, boxes=None):
        B = image.shape[0]
        image_size = (image.shape[2], image.shape[3])
        
        image_embeddings = self.image_encoder(image) 
        
        # FIX: Robustly derive M (number of independent masks to predict per image)
        # Assuming points are (B, M, N, 2) and boxes are (B, M, 4)
        if boxes is not None:
            if boxes.ndim == 2: boxes = boxes.unsqueeze(1) # Broadcast to (B, 1, 4)
            M = boxes.shape[1]
        elif points is not None:
            coords, labels = points
            if coords.ndim == 3: # Broadcast (B, N, 2) to (B, 1, N, 2)
                coords = coords.unsqueeze(1)
                labels = labels.unsqueeze(1)
            M = coords.shape[1]
            points = (coords, labels)
            
        if boxes is not None:
            boxes = boxes.reshape(B * M, -1) 
            
        if points is not None:
            coords, labels = points
            coords = coords.reshape(B * M, -1, 2) 
            labels = labels.reshape(B * M, -1)    
            points = (coords, labels)
            
        image_embeddings = image_embeddings.repeat_interleave(M, dim=0) 
        
        # FIX: Generate and provide Dense Image Positional Encodings
        image_pe = self.prompt_encoder.get_dense_pe(image_size)
        image_pe = image_pe.repeat_interleave(M, dim=0)
        
        sparse_embeddings = self.prompt_encoder(points, boxes, image_size)
        masks, iou_preds = self.mask_decoder(image_embeddings, sparse_embeddings, image_pe)
        
        masks = F.interpolate(masks, size=image_size, mode="bilinear", align_corners=False)
        
        masks = masks.reshape(B, M, self.num_masks, image_size[0], image_size[1])
        iou_preds = iou_preds.reshape(B, M, self.num_masks)
            
        return masks, iou_preds

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SAMDINOv3(model_name='dinov3_vitb16').to(device) # Swapped to ViT-B for quicker dummy testing
    model.eval()
    print("Model initialized successfully.")

    # 2. Dummy Inputs
    B, C, H, W = 2, 3, 1024, 1024
    M = 2 # Let's test with 2 distinct mask prompts per image
    dummy_image = torch.randn(B, C, H, W).to(device)

    # --- Point Prompt Setup ---
    # N points per mask prompt: coords shape (B, M, N, 2), labels shape (B, M, N)
    # 2 images, 2 target masks per image, 2 points per mask
    point_coords = torch.randint(100, 900, (B, M, 2, 2), dtype=torch.float32).to(device)
    point_labels = torch.randint(0, 2, (B, M, 2)).to(device)
    points = (point_coords, point_labels)

    # --- Box Prompt Setup ---
    # shape (B, M, 4) in format [x1, y1, x2, y2]
    boxes = torch.tensor([
        [[100, 100, 400, 400], [500, 500, 800, 800]], 
        [[200, 200, 600, 600], [150, 150, 300, 300]]
    ], dtype=torch.float32).to(device)

    # 3. Forward Pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16): # Simulating FSDP/AMP context
        with torch.no_grad():
            pred_masks, iou_preds = model(dummy_image, points=points, boxes=boxes)

    # Outputs logits of shape (B, M, 3, H, W)
    mask_probs = torch.sigmoid(pred_masks)
    print(f"Output Mask Shape: {mask_probs.shape} (Expected: {B}, {M}, 3, {H}, {W})")
    print(f"Output IoU Shape:  {iou_preds.shape} (Expected: {B}, {M}, 3)")