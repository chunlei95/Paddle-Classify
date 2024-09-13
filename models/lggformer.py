import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init

trunc_normal_ = paddle_init.TruncatedNormal(std=.02)
zeros_ = paddle_init.Constant(value=0.)
ones_ = paddle_init.Constant(value=1.)


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


class OverlapPatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    # noinspection PyTypeChecker
    def __init__(self,
                 in_channels=3,
                 out_channels=768,
                 patch_size=7,
                 stride=4):
        super().__init__()

        self.proj = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size - 1) // 2)

    def forward(self, x):
        x = self.proj(x)
        return x


class ConvNormAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm_type=None,
                 act_type=None,
                 groups=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups)
        self.norm = None
        self.act = None
        if norm_type is not None:
            self.norm = BuildNorm(out_channels, norm_type)
        if act_type is not None:
            self.act = act_type()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvStem(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 norm_type=None,
                 act_type=nn.GELU):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv_1 = ConvNormAct(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  norm_type=norm_type,
                                  act_type=act_type)
        self.conv_2 = ConvNormAct(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  norm_type=norm_type,
                                  act_type=act_type)
        self.conv_3 = ConvNormAct(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  norm_type=norm_type,
                                  act_type=act_type)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
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


class BuildNorm(nn.Layer):
    def __init__(self, channels, norm_type=None):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type is not None:
            self.norm = self.norm_type(channels)

    def forward(self, x):
        if self.norm_type is None:
            return x
        if self.norm_type == nn.LayerNorm:
            x = paddle.transpose(x, (0, 2, 3, 1))
            x = self.norm(x)
            x = paddle.transpose(x, (0, 3, 1, 2))
        else:
            x = self.norm(x)
        return x


# noinspection PyDefaultArgument,PyTypeChecker
class LGGFormer(nn.Layer):
    def __init__(self, in_channels, num_classes, patch_size=3, stage_channels=[32, 64, 128, 256],
                 encoder_stage_blocks=[3, 3, 12, 3], num_heads=[2, 4, 8, 16],
                 trans_layers=2, drop_path_rate=0.5, norm_type=nn.BatchNorm2D, act_type=nn.ReLU6):
        super().__init__()
        self.feat_channels = stage_channels
        block_list = encoder_stage_blocks
        dpr = np.linspace(0, drop_path_rate, sum(block_list)).tolist()
        encoder_dpr = dpr[0:sum(encoder_stage_blocks)]
        self.feat_channels = stage_channels
        self.stem = ConvStem(in_channels=in_channels, out_channels=stage_channels[0], norm_type=norm_type,
                             act_type=act_type)
        self.embedding = OverlapPatchEmbed(in_channels=stage_channels[0], out_channels=stage_channels[0],
                                           patch_size=patch_size, stride=2)
        self.encoder = L2GEncoder(stage_channels, encoder_stage_blocks, num_heads,
                                  trans_layers,
                                  norm_type, act_type,
                                  encoder_dpr)
        self.norm_encoder = BuildNorm(stage_channels[-1], norm_type)
        self.head = nn.Linear(stage_channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm_encoder(x)
        x = x.mean(axis=(2, 3))
        x = self.head(x)
        return x


class L2GEncoder(nn.Layer):
    def __init__(self, stage_channels, stage_blocks, num_heads, trans_layers,
                 norm_type,
                 act_type,
                 drop_path_rate):
        super().__init__()
        self.down_sample_list = nn.LayerList(
            [
                PatchCombined(dim=stage_channels[i], merge_size=3, norm_layer=norm_type)
                if i != len(stage_channels) - 1 else nn.Identity()
                for i in range(len(stage_channels))
            ]
        )
        self.l2g_layer_list = nn.LayerList(
            [
                BasicMSLLayer(stage_blocks[i], stage_channels[i], drop_path_rate[i], norm_type, act_type)
                if i < len(stage_channels) - trans_layers
                else
                BasicL2GLayer(stage_blocks[i], stage_channels[i], num_heads[i],
                              drop_path_rate[i], norm_type,
                              act_type)
                for i in range(len(stage_channels))
            ]
        )

    def forward(self, x):
        for down, l2g in zip(self.down_sample_list, self.l2g_layer_list):
            x = l2g(x)
            if type(down) != nn.Identity:
                x = down(x)
        return x


class MSLBlock(nn.Layer):
    def __init__(self, channels, norm_type, act_type, attn_drop):
        super().__init__()
        self.norm = BuildNorm(channels, norm_type)
        self.norm_mlp = BuildNorm(channels, norm_type)
        scale_channels = channels // 1
        self.conv_1 = ConvNormAct(scale_channels, scale_channels, 3, 1, 1, norm_type, act_type,
                                  groups=scale_channels)
        self.conv_2 = ConvNormAct(scale_channels, scale_channels, 3, 1, 1, norm_type, act_type,
                                  groups=scale_channels)
        self.conv_3 = ConvNormAct(scale_channels, scale_channels, 3, 1, 1, norm_type, act_type,
                                  groups=scale_channels)
        self.proj = nn.Conv2D(channels, channels, 1)
        self.drop = DropPath(attn_drop)
        self.layer_scale_1 = self.create_parameter([channels, 1, 1],
                                                   default_initializer=nn.initializer.Constant(1.0e-6))
        self.layer_scale_2 = self.create_parameter([channels, 1, 1],
                                                   default_initializer=nn.initializer.Constant(1.0e-6))
        self.mlp = Mlp(channels, channels * 4, channels, act_type, attn_drop)
        # self.mlp = AMCMixer(channels, act_type)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xl_1 = self.conv_1(x)
        xl_2 = self.conv_2(xl_1)
        xl_3 = self.conv_3(xl_2)
        xl = xl_1 + xl_2 + xl_3
        xl = self.proj(xl)
        x = x * xl
        x = self.drop(x * self.layer_scale_1) + residual
        residual = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = self.drop(x * self.layer_scale_2) + residual
        return x


class L2GBlock(nn.Layer):
    def __init__(self, channels, norm_type, act_type, num_head, attn_drop):
        super().__init__()
        self.norm = BuildNorm(channels, norm_type)
        self.norm_mlp = BuildNorm(channels, norm_type)
        scale_channels = channels // 2
        self.conv_1 = ConvNormAct(scale_channels, scale_channels, 3, 1, 1, norm_type, act_type,
                                  groups=scale_channels)
        self.conv_2 = ConvNormAct(scale_channels, scale_channels, 3, 1, 1, norm_type, act_type,
                                  groups=scale_channels)
        self.conv_3 = ConvNormAct(scale_channels, scale_channels, 3, 1, 1, norm_type, act_type,
                                  groups=scale_channels)

        self.gfem = GFEM(scale_channels, num_head, attn_drop)
        # self.gfem = FocusedLinearAttention(scale_channels, num_head, attn_drop)
        # self.gfem = SwinTransformerBlock(scale_channels, input_resolution, num_head, drop_path=drop_path_rate)
        self.proj = nn.Conv2D(scale_channels, scale_channels, 1)
        self.mod_xl = nn.Conv2D(scale_channels, scale_channels, 1)
        self.drop = DropPath(attn_drop)
        self.mlp = Mlp(channels, channels * 4, channels, act_type)
        self.layer_scale_1 = self.create_parameter([channels, 1, 1],
                                                   default_initializer=nn.initializer.Constant(1.0e-6))
        self.layer_scale_2 = self.create_parameter([channels, 1, 1],
                                                   default_initializer=nn.initializer.Constant(1.0e-6))
        # self.mlp = AMCMixer(channels, act_type)

    def forward(self, x, pre_attn):
        residual = x
        x = self.norm(x)
        x1, x2 = paddle.split(x, 2, 1)
        xl_1 = self.conv_1(x1)
        xl_2 = self.conv_2(xl_1)
        xl_3 = self.conv_3(xl_1)
        xl = xl_1 + xl_2 + xl_3
        xl = self.proj(xl)
        xg, pre_attn = self.gfem(x2, xl, pre_attn)
        mod_xl = self.mod_xl(xg)
        xl = xl * mod_xl
        x = paddle.concat([xl, xg], 1)
        x = self.drop(x * self.layer_scale_1) + residual
        residual = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = self.drop(x * self.layer_scale_2) + residual
        return x, pre_attn


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


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class AMCMixer(nn.Layer):
    def __init__(self, chans, act_type):
        super().__init__()
        self.linear_1 = nn.Conv2D(chans, chans * 4, 1)
        self.aap = nn.AdaptiveAvgPool2D(1)
        self.amp = nn.AdaptiveMaxPool2D(1)
        self.aap_act_1 = act_type()
        self.aap_act_2 = nn.Sigmoid()
        self.linear_2 = nn.Conv2D(chans * 4, chans, 1)
        self.drop = nn.Dropout2D(0.5)
        self.proj = nn.Conv2D(chans, chans, 1)

    def forward(self, x):
        mod = x
        x1 = self.aap(x)
        x2 = self.amp(x)
        x = x1 + x2
        x = self.linear_1(x)
        x = self.aap_act_1(x)
        x = self.linear_2(x)
        x = self.aap_act_2(x)
        x = self.drop(x)
        x = x * mod
        x = self.proj(x)
        return x


class BasicL2GLayer(nn.Layer):
    def __init__(self, blocks, channels, num_head, drop_path_rate, norm_type,
                 act_type):
        super().__init__()

        self.block_list = nn.LayerList(
            [L2GBlock(channels, norm_type, act_type, num_head, drop_path_rate)
             for _ in range(blocks)])

    def forward(self, x):
        pre_attn = None
        for block in self.block_list:
            x, pre_attn = block(x, pre_attn)
        return x


class BasicMSLLayer(nn.Layer):
    def __init__(self, blocks, channels, drop_rate, norm_type, act_type):
        super().__init__()
        self.block_list = nn.LayerList(
            [MSLBlock(channels, norm_type, act_type, drop_rate) for _ in range(blocks)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x


class FocusedLinearAttention(nn.Layer):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., focusing_factor=3, kernel_size=5,
                 norm_type=nn.LayerNorm, act_type=nn.GELU):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        # self.q = nn.Conv2D(dim, dim, 1)
        self.q_down = nn.Sequential(
            nn.Conv2D(dim, dim, 3, 2, 1, groups=dim),
            # BuildNorm(dim, norm_type),
            # nn.Conv2D(dim, dim, 1)
        )
        self.k = nn.Sequential(
            nn.Conv2D(dim, dim, 3, 1, 1, groups=dim),
            # BuildNorm(dim, norm_type),
            # nn.Conv2D(dim, dim, 1)
        )
        self.v = nn.Sequential(
            nn.Conv2D(dim, dim, 3, 1, 1, groups=dim),
            # BuildNorm(dim, norm_type),
            # nn.Conv2D(dim, dim, 1)
        )
        self.cpe = ConditionalPositionEncoding(dim, 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2D(dim, dim, 1)
        self.proj_drop = nn.Dropout2D(proj_drop)
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2D(in_channels=dim, out_channels=dim, kernel_size=3, groups=dim, padding=1)
        self.scale = paddle.static.create_parameter([1, 1, dim], dtype='float32')

    def forward(self, x, Q, pre_attn):
        # x = self.cpe(x)
        # residual = x
        B, C, H, W = x.shape
        q = self.q_down(Q)
        h, w = q.shape[2:]
        q = paddle.flatten(q, 2).transpose((0, 2, 1))
        k = self.k(x)
        v = self.v(x)
        v1 = v
        k = paddle.flatten(k, 2).transpose((0, 2, 1))
        v = paddle.flatten(v, 2).transpose((0, 2, 1))
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(axis=-1, keepdim=True)
        k_norm = k.norm(axis=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(axis=-1, keepdim=True)) * q_norm
        k = (k / k.norm(axis=-1, keepdim=True)) * k_norm
        q, k, v = (
            paddle.reshape(m, (B, -1, self.num_heads, self.dim // self.num_heads)).transpose((0, 2, 1, 3))
            for m in [q, k, v])
        attn = (k.transpose((0, 1, 3, 2)) @ v)
        if pre_attn is not None:
            attn = attn + pre_attn
        attn = self.attn_drop(attn)
        x = q @ attn
        x = paddle.transpose(x, (0, 2, 1, 3)).reshape((B, -1, C))
        x = paddle.transpose(x, (0, 2, 1)).reshape((B, C, h, w))
        if h != H:
            x = F.interpolate(x, [H, W])
        feature_map = self.dwc(v1)
        x = x + feature_map
        x = self.proj(x)
        # x = x + residual
        return x, attn


# noinspection PyProtectedMember,PyMethodMayBeStatic
class GFEM(nn.Layer):
    def __init__(self, channels, num_head, attn_drop, norm_type=nn.LayerNorm, act_type=nn.GELU):
        super().__init__()
        assert channels % num_head == 0
        # self.down = nn.Sequential(
        #     nn.Conv2D(channels, channels, kernel_size=3, stride=2, padding='same', groups=channels),
        # )
        # self.k = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)
        # self.v = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)
        self.down = ConvNormAct(channels, channels, 3, 2, 1, norm_type, None,
                                groups=channels)
        self.k = ConvNormAct(channels, channels, 3, 1, 1, norm_type, None,
                             groups=channels)
        self.v = ConvNormAct(channels, channels, 3, 1, 1, norm_type, None,
                             groups=channels)
        self.num_head = num_head
        self.head_dim = channels // num_head
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2D(channels, channels, 1)
        self.mlp = Mlp(channels, channels * 4, channels, act_type, attn_drop)
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
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, Q, pre_attn):
        ori_H, ori_W = x.shape[2:]
        residual = x
        Q = self.down(Q)
        B, C, H, W = Q.shape
        K = self.k(x)
        V = self.v(x)
        Q = paddle.flatten(Q, 2).transpose((0, 2, 1))
        K = paddle.flatten(K, 2).transpose((0, 2, 1))
        V = paddle.flatten(V, 2).transpose((0, 2, 1))
        B, N, C = Q.shape
        Q = paddle.reshape(Q, (B, N, self.num_head, self.head_dim)).transpose((0, 2, 1, 3))
        K = paddle.reshape(K, (B, -1, self.num_head, self.head_dim)).transpose((0, 2, 1, 3))
        V = paddle.reshape(V, (B, -1, self.num_head, self.head_dim)).transpose((0, 2, 1, 3))
        attn = (Q @ K.transpose([0, 1, 3, 2])) * self.scale
        if pre_attn is not None:
            attn = attn + pre_attn
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        xg = (attn @ V)
        xg = xg.transpose([0, 2, 1, 3]).flatten(2).transpose((0, 2, 1)).reshape((B, -1, H, W))
        xg = F.interpolate(xg, [ori_H, ori_W])
        x = self.proj(xg)
        x = x + residual
        return x, attn


if __name__ == '__main__':
    model = LGGFormer(in_channels=3, stage_channels=[96, 192, 384, 768], num_classes=19,
                      encoder_stage_blocks=[3, 3, 12, 3], patch_size=3)
    x = paddle.randn((2, 3, 224, 224))

    # calculate_flops_and_params(x, model)
    out = model(x)
    print(out.shape)
