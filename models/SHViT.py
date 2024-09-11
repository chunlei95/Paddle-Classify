import math
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant

from utils.model_utils import calculate_flops_and_params

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, act_type=None):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias_attr=False)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = None
        if act_type is not None:
            self.act = act_type()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DepthConvBNAct(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, act_type=None):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = None
        if act_type is not None:
            self.act = act_type()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class LargeStrideOverlapPatchEmbedding(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = ConvBNAct(in_channels, out_channels, 3, 2, 1)
        self.conv_2 = ConvBNAct(out_channels, out_channels, 3, 2, 1)
        self.conv_3 = ConvBNAct(out_channels, out_channels, 3, 2, 1)
        self.conv_4 = nn.Conv2D(out_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x


class FFN(nn.Layer):
    def __init__(self, dim, dim_ratio=2, drop_rate=0.):
        super().__init__()
        self.conv_1 = nn.Conv2D(dim, dim * dim_ratio, 1)
        self.act = nn.ReLU6()
        self.conv_2 = nn.Conv2D(dim * dim_ratio, dim, 1)
        self.drop = nn.Dropout2D(drop_rate)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv_2(x)
        x = self.drop(x)
        return x


class SHAttention(nn.Layer):
    def __init__(self, dim, attn_ratio, attn_drop):
        super().__init__()
        self.dim = dim
        self.attn_dim = int(self.dim * attn_ratio)
        self.qkv = nn.Sequential(
            nn.Conv2D(self.attn_dim, self.attn_dim * 3, 3, padding=1, groups=self.attn_dim),
            nn.BatchNorm2D(self.attn_dim * 3)
        )
        self.drop = nn.Dropout(attn_drop)
        self.num_head = 1
        self.scale = self.attn_dim ** 0.5
        self.proj = nn.Conv2D(dim, dim, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        x_a, x_i = paddle.split(x, (self.attn_dim, self.dim - self.attn_dim), axis=1)
        qkv = self.qkv(x_a)
        qkv = paddle.flatten(qkv, 2).transpose((0, 2, 1))
        B, N, C = qkv.shape
        qkv = paddle.reshape(qkv, (B, N, 3, C // 3)).transpose((0, 2, 1, 3))
        q, k, v = paddle.split(qkv, 3, axis=1)  # B, 1, N, C
        attn = q @ paddle.transpose(k, (0, 1, 3, 2))
        attn = attn * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.drop(attn)
        x_a = (attn @ v).transpose((0, 1, 3, 2)).reshape((B, self.attn_dim, H, W))
        x = paddle.concat([x_a, x_i], axis=1)
        x = self.proj(x)
        x = self.drop(x)
        return x


class ConvStageBlock(nn.Layer):
    def __init__(self, dim, ffn_ratio, drop_rate, drop_path_rate):
        super().__init__()
        self.drop_path = DropPath(drop_path_rate)
        self.conv = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.bn = nn.BatchNorm2D(dim)
        self.ffn_norm = nn.BatchNorm2D(dim)
        self.ffn = FFN(dim, ffn_ratio, drop_rate)
        self.layer_scale_1 = self.create_parameter([dim, 1, 1], default_initializer=Constant(value=1e-6))
        self.layer_scale_2 = self.create_parameter([dim, 1, 1], default_initializer=Constant(value=1e-6))

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.conv(x)
        x = self.drop_path(x) * self.layer_scale_1 + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.drop_path(x) * self.layer_scale_2 + residual
        return x


class SHViTBlock(nn.Layer):
    def __init__(self, dim, dim_ratio, stage_index, ffn_ratio, drop_rate, attn_drop, drop_path_rate):
        super().__init__()
        self.stage_index = stage_index
        self.drop_path = DropPath(drop_path_rate)
        self.conv = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.bn = nn.BatchNorm2D(dim)
        self.layer_scale_1 = self.create_parameter([dim, 1, 1], default_initializer=Constant(value=1e-6))
        if self.stage_index != 0:
            self.attn = SHAttention(dim, dim_ratio, attn_drop)
            self.ln = partial(nn.LayerNorm, epsilon=1e-6)(dim)
            self.layer_scale_2 = self.create_parameter([dim, 1, 1], default_initializer=Constant(value=1e-6))
        self.ffn_norm = nn.BatchNorm2D(dim)
        self.ffn = FFN(dim, ffn_ratio, drop_rate)
        self.layer_scale_3 = self.create_parameter([dim, 1, 1], default_initializer=Constant(value=1e-6))

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.conv(x)
        x = self.drop_path(x) * self.layer_scale_1 + residual

        if self.stage_index != 0:
            residual = x
            x = paddle.transpose(x, (0, 2, 3, 1))
            x = self.ln(x)
            x = paddle.transpose(x, (0, 3, 1, 2))
            x = self.attn(x)
            x = self.drop_path(x) * self.layer_scale_2 + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.drop_path(x) * self.layer_scale_3 + residual
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SEModule(nn.Layer):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(self,
                 channels,
                 rd_ratio=1. / 16,
                 rd_channels=None,
                 rd_divisor=8,
                 add_maxpool=False,
                 bias=True,
                 norm_layer=None):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = _make_divisible(channels * rd_ratio, rd_divisor, min_value=0.)
        self.fc1 = nn.Conv2D(channels, rd_channels, kernel_size=1, bias_attr=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2D(rd_channels, channels, kernel_size=1, bias_attr=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class ResidualPatchMerging(nn.Layer):
    def __init__(self, dim, out_dim, ffn_ratio, ffn_drop, drop_path_rate):
        super().__init__()
        self.block_1 = ConvStageBlock(dim, ffn_ratio, ffn_drop, drop_path_rate)
        hid_dim = int(dim * 4)
        self.conv1 = ConvBNAct(dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = ConvBNAct(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SEModule(hid_dim, .25)
        self.conv3 = ConvBNAct(hid_dim, out_dim, 1, 1, 0)
        self.block_2 = ConvStageBlock(out_dim, ffn_ratio, ffn_drop, drop_path_rate)

    def forward(self, x):
        x = self.block_1(x)
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        x = self.block_2(x)
        return x


class StageLayer(nn.Layer):
    def __init__(self, in_channels, dim_ratio, stage_index, ffn_ratio, drop_rate, attn_drop, drop_path_rate, depth):
        super().__init__()
        self.blocks = nn.LayerList([
            SHViTBlock(dim=in_channels,
                       dim_ratio=dim_ratio,
                       stage_index=stage_index,
                       ffn_ratio=ffn_ratio,
                       drop_rate=drop_rate,
                       attn_drop=attn_drop,
                       drop_path_rate=drop_path_rate[i])
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SHViT(nn.Layer):
    """
    CVPR 2024 SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design
    在实际应用中效果不行（不排除参数没调好，但概率很低）
    """
    def __init__(self,
                 in_channels,
                 num_classes,
                 stage_channels,
                 stage_depths,
                 dim_ratios,
                 ffn_ratios,
                 drop_rates,
                 attn_drops,
                 drop_path_rate):
        super().__init__()
        self.stages = len(stage_channels)
        self.patch_embed = LargeStrideOverlapPatchEmbedding(in_channels=in_channels, out_channels=stage_channels[0])
        dpr = [d for d in paddle.linspace(0, drop_path_rate, sum(stage_depths))]
        self.stage_layers = nn.LayerList(
            [
                StageLayer(in_channels=stage_channels[i],
                           dim_ratio=dim_ratios[i],
                           stage_index=i,
                           ffn_ratio=ffn_ratios[i],
                           drop_rate=drop_rates[i],
                           attn_drop=attn_drops[i],
                           drop_path_rate=dpr[sum(stage_depths[:i]):sum(stage_depths[:i]) + stage_depths[i]],
                           depth=stage_depths[i])
                for i in range(self.stages)
            ]
        )
        self.down_samples = nn.LayerList(
            [
                ResidualPatchMerging(dim=stage_channels[i],
                                     ffn_ratio=ffn_ratios[i],
                                     ffn_drop=drop_rates[i],
                                     out_dim=stage_channels[i + 1],
                                     drop_path_rate=0.)
                if i != self.stages - 1 else nn.Identity()
                for i in range(self.stages)
            ]
        )
        self.norm = nn.BatchNorm2D(stage_channels[-1])
        self.head = nn.Linear(stage_channels[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            m.weight.set_value(
                paddle.normal(
                    std=math.sqrt(2.0 / fan_out), shape=m.weight.shape))
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        for i in range(self.stages):
            x = self.stage_layers[i](x)
            x = self.down_samples[i](x)
        x = self.norm(x)
        x = paddle.mean(x, axis=(2, 3))
        x = self.head(x)
        return x


def SHViT_S1(**kwargs):
    return SHViT(stage_channels=[128, 224, 320],
                 stage_depths=[2, 4, 5],
                 dim_ratios=[1 / 4.67, 1 / 4.67, 1 / 4.67],
                 ffn_ratios=[2, 2, 2],
                 drop_rates=[0., 0., 0.],
                 attn_drops=[0., 0., 0.],
                 drop_path_rate=0.1,
                 **kwargs)


def SHViT_S2(**kwargs):
    return SHViT(stage_channels=[128, 308, 448],
                 stage_depths=[2, 4, 5],
                 dim_ratios=[1 / 4.67, 1 / 4.67, 1 / 4.67],
                 ffn_ratios=[2, 2, 2],
                 drop_rates=[0., 0., 0.],
                 attn_drops=[0., 0., 0.],
                 drop_path_rate=0.1,
                 **kwargs)


def SHViT_S3(**kwargs):
    return SHViT(stage_channels=[192, 352, 448],
                 stage_depths=[3, 5, 5],
                 dim_ratios=[1 / 4.67, 1 / 4.67, 1 / 4.67],
                 ffn_ratios=[2, 2, 2],
                 drop_rates=[0.1, 0.1, 0.1],
                 attn_drops=[0.1, 0.1, 0.1],
                 drop_path_rate=0.1,
                 **kwargs)


def SHViT_S4(**kwargs):
    return SHViT(stage_channels=[224, 336, 448],
                 stage_depths=[4, 7, 6],
                 dim_ratios=[1 / 4.67, 1 / 4.67, 1 / 4.67],
                 ffn_ratios=[2, 2, 2],
                 drop_rates=[0.1, 0.1, 0.1],
                 attn_drops=[0.1, 0.1, 0.1],
                 drop_path_rate=0.1,
                 **kwargs)


if __name__ == '__main__':
    model = SHViT_S1(in_channels=3, num_classes=19)
    x = paddle.randn((2, 3, 224, 224))
    calculate_flops_and_params(x, model)
