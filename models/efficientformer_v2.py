import itertools
import math
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Normal, Constant

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

from models.vision_transformer import to_2tuple, trunc_normal_, DropPath

EfficientFormer_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],  # 26m 83.3%
    'S2': [4, 4, 12, 8],  # 12m
    'S1': [3, 3, 9, 6],  # 79.0
    'S0': [2, 2, 6, 4],  # 75.7
}

# 26m
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

# 12m
expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

# 6.1m
expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

# 3.5m
expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}


class Attention4D(nn.Layer):
    def __init__(self,
                 dim=384,
                 key_dim=32,
                 num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU,
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(nn.Conv2D(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                                             nn.BatchNorm2D(dim), )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.q = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.k = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2D(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2D(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2D(self.num_heads * self.d), )
        self.talking_head1 = nn.Conv2D(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2D(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(act_layer(),
                                  nn.Conv2D(self.dh, dim, 1),
                                  nn.BatchNorm2D(dim), )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter([num_heads, len(attention_offsets)],
                                                      default_initializer=Constant(0.))
        self.register_buffer('attention_bias_idxs',
                             paddle.to_tensor(idxs, dtype=paddle.int64).reshape((N, N)))
        self.ab = self.attention_biases[:, self.attention_bias_idxs]

    # @paddle.no_grad()
    # def train(self, mode=True):
    #     if mode:
    #         super().train()
    #     else:
    #         super().eval()
    #     if mode and hasattr(self, 'ab'):
    #         del self.ab
    #     else:
    #         self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x).flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 3, 2))
        k = self.k(x).flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 2, 3))
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 3, 2))

        # attn = (
        #         (q @ k) * self.scale
        #         +
        #         (self.attention_biases[:, self.attention_bias_idxs]
        #          if self.training else self.ab)
        # )

        attn = ((q @ k) * self.scale + (self.attention_biases[:, self.attention_bias_idxs]))

        # attn = (q @ k) * self.scale
        attn = self.talking_head1(attn)
        attn = F.softmax(attn, axis=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)

        out = x.transpose((0, 1, 3, 2)).reshape((B, self.dh, self.resolution, self.resolution)) + v_local
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out


def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2D(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(out_chs // 2),
        act_layer(),
        nn.Conv2D(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(out_chs),
        act_layer(),
    )


class LGQuery(nn.Layer):
    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.resolution1 = resolution1
        self.resolution2 = resolution2
        self.pool = nn.AvgPool2D(1, 2, 0)
        self.local = nn.Sequential(nn.Conv2D(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
                                   )
        self.proj = nn.Sequential(nn.Conv2D(in_dim, out_dim, 1),
                                  nn.BatchNorm2D(out_dim), )

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(nn.Layer):
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 out_dim=None,
                 act_layer=None,
                 ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        self.resolution = resolution

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim
        self.resolution2 = math.ceil(self.resolution / 2)
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.resolution, self.resolution2)

        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2

        self.k = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2D(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2D(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2D(self.num_heads * self.d), )

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2D(self.dh, self.out_dim, 1),
            nn.BatchNorm2D(self.out_dim), )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(
            range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter([num_heads, len(attention_offsets)],
                                                      default_initializer=Constant(0.))
        self.register_buffer('attention_bias_idxs', paddle.to_tensor(idxs, dtype=paddle.int64).reshape((N_, N)))
        # self.ab = self.attention_biases[:, self.attention_bias_idxs]

    # @paddle.no_grad()
    # def train(self, mode=True):
    #     if mode:
    #         super().train()
    #     else:
    #         super().eval()
    #     if mode and hasattr(self, 'ab'):
    #         del self.ab
    #     else:
    #         self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.q(x).flatten(2).reshape((B, self.num_heads, -1, self.N2)).transpose((0, 1, 3, 2))
        k = self.k(x).flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 2, 3))
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 3, 2))

        # attn = (
        #         (q @ k) * self.scale
        #         +
        #         (self.attention_biases[:, self.attention_bias_idxs]
        #          if self.training else self.ab)
        # )
        attn = ((q @ k) * self.scale + (self.attention_biases[:, self.attention_bias_idxs]))

        # attn = (q @ k) * self.scale
        attn = F.softmax(attn, axis=-1)
        x = (attn @ v).transpose((0, 1, 3, 2))
        out = x.reshape((B, self.dh, self.resolution2, self.resolution2)) + v_local

        out = self.proj(out)
        return out


class Embedding(nn.Layer):
    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2D,
                 light=False, asub=False, resolution=None, act_layer=nn.ReLU, attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2D(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2D(in_chans),
                nn.Hardswish(),
                nn.Conv2D(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2D(in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2D(embed_dim)
            )
        elif self.asub:
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim,
                                   resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class Mlp(nn.Layer):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        if self.mid_conv:
            self.mid = nn.Conv2D(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2D(hidden_features)

        self.norm1 = nn.BatchNorm2D(hidden_features)
        self.norm2 = nn.BatchNorm2D(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class AttnFFN(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 act_layer=nn.ReLU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):

        super().__init__()

        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = self.create_parameter([dim, 1, 1],
                                                       default_initializer=Constant(layer_scale_init_value))
            self.layer_scale_2 = self.create_parameter([dim, 1, 1],
                                                       default_initializer=Constant(layer_scale_init_value))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))

        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class FFN(nn.Layer):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = self.create_parameter([dim, 1, 1],
                                                       default_initializer=Constant(layer_scale_init_value))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


# noinspection PyTypeChecker
def eformer_block(dim,
                  index,
                  layers,
                  pool_size=3,
                  mlp_ratio=4.,
                  act_layer=nn.GELU,
                  norm_layer=nn.LayerNorm,
                  drop_rate=.0,
                  drop_path_rate=0.,
                  use_layer_scale=True,
                  layer_scale_init_value=1e-5,
                  vit_num=1,
                  resolution=7,
                  e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            if index == 2:
                stride = 2
            else:
                stride = None
            blocks.append(
                AttnFFN(dim,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        drop=drop_rate, drop_path=block_dpr,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                        resolution=resolution,
                        stride=stride)
            )
        else:
            blocks.append(
                FFN(dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate, drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value)
            )
    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormerV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 layers,
                 embed_dims=None,
                 mlp_ratios=4,
                 downsamples=None,
                 pool_size=3,
                 norm_layer=nn.BatchNorm2D,
                 act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 fork_feat=False,
                 vit_num=0,
                 distillation=False,
                 resolution=224,
                 e_ratios=expansion_ratios_L):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = stem(in_channels, embed_dims[0], act_layer=act_layer)

        network = []
        for i in range(len(layers)):
            stage = eformer_block(embed_dims[i],
                                  i,
                                  layers,
                                  pool_size=pool_size,
                                  mlp_ratio=mlp_ratios,
                                  act_layer=act_layer,
                                  norm_layer=norm_layer,
                                  drop_rate=drop_rate,
                                  drop_path_rate=drop_path_rate,
                                  use_layer_scale=use_layer_scale,
                                  layer_scale_init_value=layer_scale_init_value,
                                  resolution=math.ceil(resolution / (2 ** (i + 2))),
                                  vit_num=vit_num,
                                  e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                if i >= 2:
                    asub = True
                else:
                    asub = False
                network.append(
                    Embedding(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        resolution=math.ceil(resolution / (2 ** (i + 2))),
                        asub=asub,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                    )
                )

        self.network = nn.LayerList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        # print(x.size())
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out


def EfficientFormerV2_S0(**kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S0'],
        embed_dims=EfficientFormer_width['S0'],
        downsamples=[True, True, True, True, True],
        vit_num=2,
        drop_path_rate=0.0,
        e_ratios=expansion_ratios_S0,
        **kwargs)
    return model


def EfficientFormerV2_S1(**kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S1'],
        embed_dims=EfficientFormer_width['S1'],
        downsamples=[True, True, True, True],
        vit_num=2,
        drop_path_rate=0.0,
        e_ratios=expansion_ratios_S1,
        **kwargs)
    return model


def EfficientFormerV2_S2(**kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S2'],
        embed_dims=EfficientFormer_width['S2'],
        downsamples=[True, True, True, True],
        vit_num=4,
        drop_path_rate=0.2,
        e_ratios=expansion_ratios_S2,
        **kwargs)
    return model


def EfficientFormerV2_L(**kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['L'],
        embed_dims=EfficientFormer_width['L'],
        downsamples=[True, True, True, True],
        vit_num=6,
        drop_path_rate=0.1,
        e_ratios=expansion_ratios_L,
        **kwargs)
    return model
