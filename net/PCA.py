import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_atten+ x) 
        x_out = self.proj_drop(x_out)

        return x_out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.sr_ratio = sr_ratio
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)                    
        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        
        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x1, x2, H, W):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape
        assert B1 == B2 and C1 == C2 and N1 == N2, "x1 and x2 should have the same dimensions" 

        q1 = self.q1(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3) 

        if self.sr_ratio > 1:
            x2_ = x2.permute(0, 2, 1).reshape(B2, C2, H, W) 
            x2_ = self.sr(x2_).reshape(B2, C2, -1).permute(0, 2, 1) 
            x2_ = self.norm(x2_)
            kv2 = self.kv2(x2_).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv2 = self.kv2(x2).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
    
        k2, v2 = kv2[0], kv2[1]

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v2).transpose(1, 2).reshape(B2, N2, C2)
        x_out = self.proj(x_atten+x1)
        x_out = self.proj_drop(x_out)

        return x_out

class PCASC(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=8):
        super(PCASC, self).__init__()                         
        self.SA_x1 = SelfAttention(dim, num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CA_x1toX2 = CrossAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.SA_x2 = SelfAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CA_x2toX1 = CrossAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.proj = nn.Linear(dim, dim)

        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "x1 and x2 should have the same dimensions"
        x1_flat = x1.flatten(2).transpose(1, 2)  
        x2_flat = x2.flatten(2).transpose(1, 2)  
        x1_self_enhance =  self.SA_x1(x1_flat,H1, W1)
        x2_cross_enhance = self.CA_x1toX2(x2_flat,x1_self_enhance,H1, W1)
        x2_self_enhance = self.SA_x2(x2_cross_enhance,H1, W1)
        x1_cross_enhance = self.CA_x2toX1(x1_self_enhance,x2_self_enhance,H1, W1)  
        Fuse = self.proj(x1_cross_enhance)   

        Fuse_out = Fuse.permute(0, 2, 1).reshape(B1, C1, H1, W1).contiguous()
          
        return Fuse_out  