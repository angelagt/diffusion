import math
import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn

from .attention import QKVAttention, AttentionBlock
from .resblock import ResBlock
from .timestep import timestep_embedding
from .utils import (
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    convert_module_to_f16,
    convert_module_to_fp32,
)
from .modules import SiLU, linear, TimestepBlock, TimestepEmbedSequential


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class UNetModel(nn.Module):
    """
    The full UNet model with attention, timestep embedding, and multi-label conditioning.
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Multi-label conditioning projection
        if self.num_classes is not None:
            self.label_proj = nn.Linear(self.num_classes, time_embed_dim)

        # Build U-Net blocks (unchanged)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_fp32)
        self.middle_block.apply(convert_module_to_fp32)
        self.output_blocks.apply(convert_module_to_fp32)

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        :param x: [N x C x ...]
        :param timesteps: [N]
        :param y: [N x num_classes] multi-hot labels
        """
        assert (y is not None) == (self.num_classes is not None), \
            "Must specify y iff model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0], self.num_classes), \
                f"y must be (B, {self.num_classes}), got {tuple(y.shape)}"
            emb = emb + self.label_proj(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Like forward, but returns intermediate features.
        """
        assert (y is not None) == (self.num_classes is not None)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0], self.num_classes)
            emb = emb + self.label_proj(y)

        result = {'down': [], 'up': []}
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            result['down'].append(h.type(x.dtype))
        result['middle'] = self.middle_block(h, emb).type(x.dtype)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, result['down'].pop()], dim=1)
            h = module(cat_in, emb)
            result['up'].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        new_h, new_w = x.shape[2:]
        up = F.interpolate(low_res, (new_h, new_w), mode='bilinear')
        return super().forward(th.cat([x, up], dim=1), timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        new_h, new_w = x.shape[2:]
        up = F.interpolate(low_res, (new_h, new_w), mode='bilinear')
        return super().get_feature_vectors(th.cat([x, up], dim=1), timesteps, **kwargs)
```
