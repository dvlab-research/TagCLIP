
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
from timm.models.layers import trunc_normal_

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        '''
        Args:
            ATM block, relationship_descriptor
            xq: [2, 15, 512],   [2, 171, 512]    
            xk: [2, 1024, 512], [2, 171, 512]
            xv: [2, 1024, 512], [2, 171, 512]
        '''

        B, Nq, C = xq.size() # 1, 21, 512
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        # q: [2, 8, 15, 64],    [2, 8, 171, 64]
        # k: [2, 8, 1024, 64],  [2, 8, 171, 64]
        # v: [2, 8, 1024, 64],  [2, 8, 171, 64]
        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        # import pdb;pdb.set_trace()
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        # attn: [2, 8, 15, 1024], [2, 8, 171, 171]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x: [2, 15, 512], [2, 171, 512]
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads

if __name__ == '__main__':
    attention = Attention(dim=512, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.0)
    print(attention)

    # xq = torch.randn([2, 15, 512])
    # xk = torch.randn([2, 1024, 512])
    # xv = torch.randn([2, 1024, 512])
    # out = attention(xq, xk, xv)

    dim = 512
    C = 171
    bs = 2
    # cls_token:  [2, 171, 512]
    # text_token: [2, 171, 512]
    cls_token = torch.randn([bs, C, dim])
    text_token = torch.randn([bs, C, dim])

    # x:    [171, 2, 512]
    # attn: [2, 171, 171]
    x, attn = attention(cls_token, text_token, cls_token)

    # x:    [171, 2, 512] -> [2, 171, 512]
    x = x.transpose(0, 1)
    print(x.shape, attn.shape)
