import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (Conv does patchify + linear proj in one shot)."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.proj(x)                 # (B, D, 8, 8)
        x = x.flatten(2).transpose(1, 2) # (B, 64, D)
        return x
        
        
class TinyViT(nn.Module):
    def __init__(
        self, 
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token + positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, x):
        x = self.patch_embed(x)
        
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.encoder(x)
        x = self.norm(x)
        
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        
        return logits