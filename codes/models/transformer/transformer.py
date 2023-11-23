# --coding:utf-8--

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from functools import partial
from codes.models.transformer.trans_helper import to_2tuple, DropPath, trunc_normal_
from einops import rearrange
from codes.models.cnns.conv_depthwise_separable import Conv_DW


def GELU(x):
	return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def freeze(layer):
	'''
	冻结 layer的参数，不参与更新
	'''
	for child in layer.children():
		for param in child.parameters():
			param.requiers_grad = False


class PatchEmbed(nn.Module):
	""" Image to Patch Embedding
	"""

	def __init__(self, img_size, patch_size=16, in_chans=3, embed_dim=768):
		super().__init__()
		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)

		self.patch_size = patch_size
		assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
			f"img_size {img_size} should be divided by patch_size {patch_size}."
		self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
		self.num_patches = self.H * self.W
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
		self.norm = nn.LayerNorm(embed_dim)

		freeze(self.proj)	# 冻结该层参数

	def forward(self, x):
		B, C, H, W = x.shape
		x = self.proj(x).flatten(2).transpose(1, 2)
		x = self.norm(x)
		H, W = H // self.patch_size[0], W // self.patch_size[1]

		return x, (H, W)


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
		super().__init__()
		assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

		self.dim = dim
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		self.q = nn.Linear(dim, dim, bias=qkv_bias)
		self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		self.sr_ratio = sr_ratio
		if sr_ratio > 1:
			self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
			self.norm = nn.LayerNorm(dim)

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

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)

		return x


class SpatialAttention(nn.Module):
	'''
	空间注意力模块
	'''
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()
		assert kernel_size in (3,7), "kernel size must be 3 or 7"
		padding = 3 if kernel_size == 7 else 1

		self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avgout = torch.mean(x, dim=1, keepdim=True)     # 通道维度求平均
		maxout, _ = torch.max(x, dim=1, keepdim=True)   # 通道维度求最大值
		x = torch.cat([avgout, maxout], dim=1)
		x = self.conv(x)
		out = self.sigmoid(x) * x
		return out


class CA_wx(nn.Module):
	'''
	通道注意力模块
	'''
	def __init__(self, ch_in, rotio=16):
		super(CA_wx, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.sharedMLP = nn.Sequential(
			nn.Conv2d(ch_in, ch_in // rotio, 1, bias=False), nn.ReLU(True),
			Conv_DW(ch_in // rotio, ch_in // rotio, 3),
			nn.Conv2d(ch_in // rotio, ch_in, 1, bias=False))
		self.softmax = nn.Softmax()

		self.dw = Conv_DW(ch_in// rotio, ch_in // rotio, 3)


	def forward(self, x):
		b, c, h, w = x.size()

		avgout = self.sharedMLP(self.avg_pool(x))
		maxout = self.sharedMLP(self.max_pool(x))
		out = x * self.softmax(avgout + maxout)
		# out = out.flatten(2).transpose(1, 2)
		return out


class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


class ConvFF(nn.Module):

	def __init__(self, dim=192, scale=4, depth_kernel=3, patch_height=14, patch_width=14, dropout=0.):
		super().__init__()

		scale_dim = dim * scale
		self.up_proj = nn.Sequential(
			# Rearrange('b (h w) c -> b c h w', h=patch_height, w=patch_width),
			nn.Conv2d(dim, scale_dim, kernel_size=1),
			nn.Hardswish()
		)

		self.depth_conv = nn.Sequential(
			nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=True),
			nn.Conv2d(scale_dim, scale_dim, kernel_size=1, bias=True),
			nn.Hardswish()
		)

		self.down_proj = nn.Sequential(
			nn.Conv2d(scale_dim, dim, kernel_size=1),
			nn.Dropout(dropout),
			# Rearrange('b c h w ->b (h w) c')
		)

	def forward(self, x):
		x = self.up_proj(x)
		x = self.depth_conv(x)
		x = self.down_proj(x)
		x = rearrange(x, 'b c h w ->b (h w) c')
		return x


class Block(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
				 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1, patch_size=16):
		super().__init__()

		self.bn = nn.BatchNorm2d(dim)
		# attention
		# 1 normal attention
		self.norm1 = norm_layer(dim)
		self.attn = Attention(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
							  attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
		if drop_path > 0. : self.drop_path = DropPath(drop_path)
		self.drop_path = drop_path

		# 2 spatial attention + channel attention
		self.bn = nn.BatchNorm2d(dim)
		self.patch_size = patch_size
		self.sa = SpatialAttention(7)
		self.ca = CA_wx(dim, mlp_ratio)

		# feed forward
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.fc = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
		self.conv_fc = ConvFF(dim=dim, scale=mlp_ratio, depth_kernel=3, patch_width=patch_size, patch_height=patch_size)

	def forward(self, x, H, W,):
		x = x + self.attn(self.norm1(x), H, W)
		x = x + self.fc(self.norm2(x))
		return x


class Trans(nn.Module):
	def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dims=64,
				 num_heads=1, mlp_ratios=4, qkv_bias=False, qk_scale=None, drop_rate=0.,
				 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
				 depths=2, sr_ratios=1):
		super().__init__()
		self.depths = depths
		self.embed_dims = embed_dims
		self.img_size = img_size

		# patch_embed
		self.patch_embed1 = PatchEmbed(img_size=img_size,patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims)

		# pos_embed
		self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims))
		self.pos_drop1 = nn.Dropout(p=drop_rate)

		# transformer
		self.transformer = Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_size=img_size//patch_size,
										   drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,sr_ratio=sr_ratios)

		self.conv_up0 = nn.Conv2d(embed_dims, in_chans, kernel_size=3, stride=1, padding=1)

		# init weights
		trunc_normal_(self.pos_embed1, std=.02)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)


	def forward(self, x):
		B, C, H, W = x.shape
		identity = x

		# 如果输入图像的size与初始化patch_embed1的不同，则resize
		if (self.img_size, self.img_size) != (H, W):
			x = F.interpolate(x, size=(self.img_size, self.img_size))

		# patch embedding
		x,( H_p,W_p )= self.patch_embed1(x)
		x = x + self.pos_embed1
		x = self.pos_drop1(x)

		# transformer: attention + fc
		x = self.transformer(x, H_p, W_p,)

		out = self.up(x, H, W, H_p,W_p)     # 经过path_embed后，实际上是下采样了，下采样因子为patch_size

		return out

	def up(self, x, H, W, H_p,W_p):
		B, N, C = x.size()
		assert N == H_p * W_p
		x = x.permute(0, 2, 1)
		x = x.view(-1, C, H_p, W_p)
		# x = rearrange(x, 'b (h w) c -> b c h w', h=H_p, w=W_p)
		x = F.interpolate(x, size=(H, W))
		x = self.conv_up0(x)
		return x

