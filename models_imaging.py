import torch
import numpy as np
import torch.nn as nn

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import Mlp

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from util.trunc_normal import trunc_normal_
from skimage.metrics import structural_similarity as ssim
# from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import peak_signal_noise_ratio as psnr
from util.tflct import *
import math
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func
from functools import reduce, lru_cache
from operator import mul

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
        
        self.attn = FlashAttention(
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
    
class TransientsZipper(nn.Module):
    def __init__(self, histogram_length=300, embed_dim=128):
        super(TransientsZipper, self).__init__()
        # self.zip = nn.Linear(in_features=histogram_length, out_features=embed_dim)
        # self.unzip = nn.Linear(in_features=embed_dim, out_features=histogram_length)
        
        self.zip = nn.Sequential(
            nn.Linear(in_features=histogram_length, out_features=embed_dim),
            nn.ReLU(),
        )
        self.unzip = nn.Sequential(
            nn.Linear(embed_dim, histogram_length),  # 对应地增加解码层
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
 
class NLOSVideoFlash(nn.Module):
    def __init__(self, dim, depth, num_heads,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 use_mlp=True, use_conv=False):

        super().__init__()
        self.dim = dim
        self.depth = depth

        self.zip = nn.Sequential(
            nn.Linear(in_features=200, out_features=dim),
            nn.ReLU(),
        )
        self.unzip = nn.Sequential(
            nn.Linear(dim, 16),  # Adjust output features as needed
        )

        # Positional embeddings
        self.sequence_length = None  # Will be set in forward
        self.pos_embed = None  # Initialize in forward

        # Build Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        # Initialize weights (other layers)
        self.initialize_weights(16,16)

    def initialize_weights(self, h, w):
        # 计算序列长度
        self.sequence_length = (h * w) * 2  # 乘以2，因为有当前和前一组数据

        # 空间位置嵌入
        spatial_pos_embed = get_2d_sincos_pos_embed(self.dim, h, cls_token=False)  # 形状：(h * w, dim)
        spatial_pos_embed = torch.from_numpy(spatial_pos_embed).float()  # 转换为 PyTorch 张量

        # 时间位置嵌入
        temporal_pos_embed = get_1d_sincos_pos_embed(self.dim, 2)  # 形状：(2, dim)
        temporal_pos_embed = torch.from_numpy(temporal_pos_embed).float()  # 转换为 PyTorch 张量

        # 添加新维度以便广播
        spatial_pos_embed = spatial_pos_embed.unsqueeze(1)  # 形状：(h*w, 1, dim)
        temporal_pos_embed = temporal_pos_embed.unsqueeze(0)  # 形状：(1, 2, dim)

        # 合并空间和时间位置嵌入
        pos_embed = spatial_pos_embed + temporal_pos_embed  # 形状：(h*w, 2, dim)

        # 展平为 (sequence_length, dim)
        pos_embed = pos_embed.view(1, self.sequence_length, self.dim)  # 添加批次维度

        # 将位置嵌入赋值给模型参数
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_noise(self, x):
        b, t, h, w = x.shape
        noise_b = torch.rand_like(x) * torch.FloatTensor(1).uniform_(0.2, 0.5).item()
        total_signal = x + noise_b
        return total_signal

    def forward(self, x_current, x_previous, if_syn):
        if if_syn:
            x_current = self.add_noise(x_current)
            x_current = x_current[:, :, ::4, ::4]
            x_previous = self.add_noise(x_previous)
            x_previous = x_previous[:, :, ::4, ::4]

        b, t, h, w = x_current.shape

        # # Initialize positional embeddings (if not already done)
        # if self.pos_embed is None or self.pos_embed.shape[1] != (h * w * 2):
        #     self.initialize_weights(h, w)

        # Reshape inputs
        x_current = rearrange(x_current, 'b t h w -> b (h w) t')
        x_previous = rearrange(x_previous, 'b t h w -> b (h w) t')

        # Stack along new temporal dimension
        x_combined = torch.stack((x_current, x_current-x_previous), dim=2)  # Shape: (b, h*w, 2, t)

        # Flatten temporal dimensions
        x_combined = x_combined.view(b, h * w * 2, t)  # Shape: (b, h*w*2, t)

        # Apply zip layer
        x = self.zip(x_combined)  # Shape: (b, sequence_length, dim)

        # Add positional embeddings
        x = x + self.pos_embed  # Shape: (b, sequence_length, dim)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Unzip layer
        x = self.unzip(x)  # Shape: (b, sequence_length, output_dim)

        # Reshape back to image dimensions
        x = x.view(b, h * w, 2, -1)  # Shape: (b, h*w, 2, output_dim)
        # print(x.shape)
        # Take the output corresponding to the current time step
        x = x[:, :, 0, :]  # Shape: (b, h*w, output_dim)

        # Reshape to (batch_size, channels, height, width)
        # output = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        output = rearrange(x,'b (h w) (p0 p1 t) -> b t (h p0) (w p1)', p0=4, p1=4, b=b, h=h, w=w, t=1)
        return output, x  # Return features if needed


if __name__ == '__main__':
    # import pdb
    # pdb.set_trace()
    # model = mae_vit_base_patch16_dec512d8b()
    
    a = NLOSUpsamplingSwin(depth=8, num_heads=16, input_resolution=(8,8), dim=128, window_size=8).cuda()
    test_img = torch.zeros(4, 64*64, 128).cuda()
    b = a(test_img, (64,64))
    print(b.shape)
    print('-----------------------------')
    
    # a = MaskedAutoencoderViT().cuda()
    # test_img = torch.zeros(7, 5, 224, 224, 3).cuda()
    # b = a(test_img)
    # print(b.shape)
    # print('-----------------------------')

    # a = PatchEmbed(224, 16, 3, 768)
    # test_img = torch.zeros(337,3,224,224) 
    # b = a(test_img)
    # print(b.shape)
    # print('-----------------------------')

    # a = HistogramEmbed(16, 300, 768)
    # test_img = torch.zeros(337,16,300) 
    # b = a(test_img)
    # print(b.shape)
    # print('-----------------------------')

    # a = HistogramEmbedConv(16, 300, 768)
    # test_img = torch.zeros(337,16,300) 
    # b = a(test_img)
    # print(b.shape)
    # print('-----------------------------')

  
