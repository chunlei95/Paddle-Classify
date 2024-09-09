import functools
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class BuildNorm(nn.Layer):
    def __init__(self, channels, norm_type=None):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type is not None:
            self.norm = self.norm_type(channels)

    def forward(self, x):
        if self.norm_type is None:
            return x
        if self.norm_type == nn.LayerNorm or (type(self.norm_type) == functools.partial and self.norm_type.func == nn.LayerNorm):
            x = paddle.transpose(x, (0, 2, 3, 1))
            x = self.norm(x)
            x = paddle.transpose(x, (0, 3, 1, 2))
        else:
            x = self.norm(x)
        return x


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


@paddle.jit.not_to_static
def swapdim(x, dim1, dim2):
    a = list(range(len(x.shape)))
    a[dim1], a[dim2] = a[dim2], a[dim1]
    return x.transpose(a)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2D(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2D(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class LK(nn.Layer):
    def __init__(self,
                 channels,
                 kernel_size=7,
                 dilation=3):
        super().__init__()
        dwd_k = math.ceil(kernel_size / dilation)
        dw_k = 2 * dilation - 1
        self.dw_dilation_conv = nn.Conv2D(in_channels=channels,
                                          out_channels=channels,
                                          kernel_size=dwd_k,
                                          dilation=dilation,
                                          padding='same',
                                          groups=channels)
        self.dw_conv = nn.Conv2D(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=dw_k,
                                 padding='same',
                                 groups=channels)

    def forward(self, x):
        self.dw_dilation_conv(self.dw_conv(x))
        return x


class MSLKA(nn.Layer):
    def __init__(self, channels, kernel_sizes=[7, 11, 21], dilation=3, drop_rate=0.2):
        super().__init__()
        self.base = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)
        self.b2 = LK(channels, kernel_size=kernel_sizes[0], dilation=dilation)
        self.b3 = LK(channels, kernel_size=kernel_sizes[1], dilation=dilation)
        self.b4 = LK(channels, kernel_size=kernel_sizes[2], dilation=dilation)
        self.proj = nn.Conv2D(channels, channels, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x1 = self.base(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        mod = x1 + x2 + x3 + x4
        mod = self.proj(mod)
        mod = self.drop(mod)
        return x * mod


class MSLKA_S(nn.Layer):
    def __init__(self, channels, kernel_sizes=[7, 11, 21], dilation=3, drop_rate=0.2):
        super().__init__()
        self.mid_channels = channels // 4
        self.base = nn.Conv2D(channels - self.mid_channels * 3, channels - self.mid_channels * 3, 3, 1, 1,
                              groups=channels - self.mid_channels * 3)
        self.b2 = LK(self.mid_channels, kernel_size=kernel_sizes[0], dilation=dilation)
        self.b3 = LK(self.mid_channels, kernel_size=kernel_sizes[1], dilation=dilation)
        self.b4 = LK(self.mid_channels, kernel_size=kernel_sizes[2], dilation=dilation)
        self.proj = nn.Conv2D(channels, channels, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x1, x2, x3, x4 = paddle.split(x, [x.shape[1] - self.mid_channels * 3, self.mid_channels, self.mid_channels,
                                          self.mid_channels], axis=1)
        x1 = self.base(x1)
        x2 = self.b2(x2)
        x3 = self.b3(x3)
        x4 = self.b4(x4)
        mod = paddle.concat([x1, x2, x3, x4], axis=1)
        mod = self.proj(mod)
        mod = self.drop(mod)
        return x * mod


class Attention(nn.Layer):
    def __init__(self, d_model, drop):
        super().__init__()
        self.proj_1 = nn.Conv2D(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MSLKA_S(d_model, drop_rate=drop)
        self.proj_2 = nn.Conv2D(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.encoding = ConditionalPositionEncoding(channels=dim, encode_size=3)
        self.norm1 = nn.BatchNorm2D(dim)
        self.attn = Attention(dim, drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2D(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim,
        #                hidden_features=mlp_hidden_dim,
        #                act_layer=act_layer,
        #                drop=drop)
        self.local = LocalBlock(channels=dim, kernel_size=3)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))
        self.layer_scale_2 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

    def forward(self, x):
        x = self.encoding(x)
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.local(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = nn.BatchNorm2D(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x


class PatchCombined(nn.Layer):
    def __init__(self, dim, merge_size=3, norm_layer=None):
        super().__init__()
        self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.reduction = nn.Conv2D(4 * dim, 2 * dim, kernel_size=merge_size, padding='same', groups=dim)
        self.channel_interact = nn.Conv2D(2 * dim, 2 * dim, 1)
        self.norm = BuildNorm(4 * dim, norm_type=norm_layer)

    def forward(self, x):
        B, C, H, W = x.shape

        x = paddle.transpose(x, (0, 2, 3, 1))
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = x.transpose([0, 3, 1, 2])
            x = F.pad(x, [0, W % 2, 0, H % 2])
            x = x.transpose([0, 2, 3, 1])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.reshape([B, -1, 4 * C])  # B H/2*W/2 4*C

        x = paddle.transpose(x, (0, 3, 1, 2))
        x = self.norm(x)
        x = self.reduction(x)
        x = self.channel_interact(x)
        return x


class LocalBlock(nn.Layer):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv_1 = nn.Conv2D(channels, channels, kernel_size, 1, (kernel_size - 1) // 2, groups=channels)
        self.conv_2 = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)
        self.conv_3 = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)
        self.proj = nn.Conv2D(channels, channels, 1)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x = x1 + x2 + x3
        x = self.proj(x)
        return x


class ConditionalPositionEncoding(nn.Layer):
    def __init__(self, channels, encode_size):
        super().__init__()
        self.cpe = nn.Conv2D(in_channels=channels,
                             out_channels=channels,
                             kernel_size=encode_size,
                             padding='same',
                             groups=channels)

    def forward(self, x):
        residual = x
        x = self.cpe(x)
        x = x + residual
        return x


class Modified_VAN(nn.Layer):
    r""" VAN
    A PaddlePaddle impl of : `Visual Attention Network`  -
      https://arxiv.org/pdf/2202.09741.pdf
    """

    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 class_num=1000,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 flag=False):
        super().__init__()
        if flag == False:
            self.class_num = class_num
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x for x in paddle.linspace(0, drop_path_rate, sum(depths))
               ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                # patch_size=7 if i == 0 else 3,
                patch_size=3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i]) if i == 0 else PatchCombined(dim=embed_dims[i - 1], norm_layer=norm_layer)

            block = nn.LayerList([
                Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j]) for j in range(depths[i])
            ])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3],
                              class_num) if class_num > 0 else nn.Identity()

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

    def forward_features(self, x):
        # B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            for blk in block:
                x = blk(x)
            B, C, H, W = x.shape
            x = x.flatten(2)
            x = swapdim(x, 1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape([B, H, W, x.shape[2]]).transpose([0, 3, 1, 2])

        return x.mean(axis=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
