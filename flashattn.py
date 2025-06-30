# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import pdb
import torch
import numpy as np
import torch.nn as nn

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import Mlp
from flash_attn import flash_attn_qkvpacked_func
import math
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop).reshape(B, N, C)

        # pdb.set_trace()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class UpBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            act_layer()
        )
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, int(math.sqrt(N)), int(math.sqrt(N)))  # [B, C, H, W]
        x = self.upsample_layer(x)  # [B, C, 2H, 2W]
        x = x.flatten(2).permute(0, 2, 1)  # [B, N', C]

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        print("Here~~~~~~~~~~~~")
        print(B,C,H,W)
        print(self.proj(x).shape)
        x = self.proj(x).flatten(2).transpose(1, 2)
        print(x.shape)
        return x
    
class HistogramEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, histogram_num=16, histogram_length=300, embed_dim=768):
        super().__init__()
        self.num_patches = histogram_num
        self.histogram_num = histogram_num
        self.histogram_length = histogram_length
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_features=histogram_length, out_features=embed_dim)

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, histogram_num, histogram_length = x.shape
        # FIXME look at relaxing size constraints
        assert histogram_length == self.histogram_length and histogram_num == self.histogram_num, \
            f"Input transients size ({histogram_length}*{histogram_num}) doesn't match model ({self.histogram_length}*{self.histogram_num})."
        x = self.proj(x)
        return x

class TransientsZipper(nn.Module):
    def __init__(self, histogram_length=300, embed_dim=128):
        super(TransientsZipper, self).__init__()
        self.zip = nn.Sequential(
            nn.Linear(in_features=histogram_length, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )
        self.unzip = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, histogram_length),  # 对应地增加解码层
        )
        
    def forward(self, x):
        encoded = self.zip(x)
        decoded = self.unzip(encoded)
        return decoded
    
    def Zip(self, x):
        encoded = self.zip(x)
        return encoded  
    
    def Unzip(self, x):
        decoded = self.unzip(x)
        return decoded   
    
class NLOSUpsamplingViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, target_resolution=64, scale_factor=16, histogram_length=256,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
    ):
        """
        mask_ids: [1, num_of_ids]
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.Zipper = TransientsZipper(histogram_length, embed_dim)
        # self.num_patches = target_resolution * target_resolution
        self.num_patches = 64
        
        self.target_resolution = target_resolution
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed_upsampled = nn.Parameter(torch.zeros(1, 4096, embed_dim), requires_grad=False)         
        # Convert the result back to a PyTorch tensor
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])   
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.norm = norm_layer(embed_dim)
        
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
 
    def forward(self, imgs):

        b, t, h, w = imgs.shape
        mask = torch.ones([b,h,w], device=imgs.device)
        mask[:,::4,::4] = 0
        imgs = imgs[:,:,::4,::4]
        imgs = rearrange(imgs, 'b t h w -> b (h w) t').contiguous()
        mask = rearrange(mask, 'b h w -> b (h w)').contiguous()
        
        imgs_latent = self.Zipper.Zip(imgs)
        
        ids_shuffle = torch.argsort(mask, dim=1)  # ascend: 0 is keep, 1 is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(imgs_latent.shape[0], ids_restore.shape[1] - imgs_latent.shape[1], 1)
        
        imgs_latent = torch.cat([imgs_latent, mask_tokens], dim=1)  # no cls token
        
        imgs_latent = torch.gather(imgs_latent, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs_latent.shape[2]))  # unshuffle

        
        imgs_latent = imgs_latent + self.pos_embed_upsampled
        
        # apply Transformer blocks
        for blk in self.blocks:
            imgs_latent = blk(imgs_latent)
        
        imgs_latent = self.norm(imgs_latent)

        imgs_pred = self.Zipper.Unzip(imgs_latent)

        return rearrange(imgs_pred,'b (h w) t -> b t h w')

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':

    import time
    a = NLOSUpsamplingViT(target_resolution=64, histogram_length=512, embed_dim=256, depth=8).cuda().half()
    a.eval()
    test_img = torch.zeros(1, 512, 64, 64).cuda().half()
    start = time.time()
    for i in range(2000):
        b = a(test_img)
    end = time.time()
    print(end - start)

