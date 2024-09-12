import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant


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


class Stem(nn.Layer):
    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or out_channels
        self.conv_1 = ConvBNAct(in_channels, mid_channels, 3, 2, 1, act_type=nn.ReLU)
        self.conv_2 = ConvBNAct(mid_channels, out_channels, 3, 2, 1, act_type=nn.ReLU)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class SEBlock(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv_1 = nn.Conv2D(channels, channels // 4, 1)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.conv_2 = nn.Conv2D(channels // 4, channels, 1)
        self.act_1 = nn.ReLU()
        self.act_2 = nn.Sigmoid()

    def forward(self, x):
        a = self.pool(x)
        a = self.conv_1(a)
        a = self.act_1(a)
        a = self.conv_2(a)
        a = self.act_2(a)
        x = x * a
        return x


class DownSampling(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_repvit = RepViTBlock(in_channels, 0, 0., 0.)
        self.sp_down = ConvBNAct(in_channels, in_channels, 3, 2, 1, groups=in_channels)
        self.ch_down = nn.Conv2D(in_channels, out_channels, 1)
        self.act = nn.ReLU()
        self.ffn = FFN(out_channels, 2, 0.)

    def forward(self, x):
        x = self.pre_repvit(x)
        x = self.sp_down(x)
        x = self.ch_down(x)
        x = self.act(x)
        x = self.ffn(x)
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


class RepViTBlock(nn.Layer):
    def __init__(self, channels, block_index, drop_rate, drop_path_rate):
        super().__init__()
        self.dw_3x3 = ConvBNAct(channels, channels, 3, 1, 1, groups=channels)
        self.dw_1x1 = nn.Conv2D(channels, channels, 1, 1, 0, groups=channels)
        self.block_index = block_index
        if self.block_index % 2 == 0:
            self.se_block = SEBlock(channels)
        self.norm_conv = nn.BatchNorm2D(channels)
        self.norm_ffn = nn.BatchNorm2D(channels)
        self.ffn = FFN(channels, 2, drop_rate)
        self.drop_path = DropPath(drop_path_rate)
        self.drop = nn.Dropout(0.2)
        self.layer_scale = self.create_parameter([channels, 1, 1], default_initializer=Constant(value=1.0e-6))

    def forward(self, x):
        # if self.training:
        #     residual = x
        #     x = self.norm_conv(x)
        #     x1 = self.dw_3x3(x)
        #     x2 = self.dw_1x1(x)
        #     x = x1 + x2 + residual
        # else:
        #     pass
        residual = x
        x = self.norm_conv(x)
        x1 = self.dw_3x3(x)
        x2 = self.dw_1x1(x)
        x = x1 + x2
        x = x + residual

        if self.block_index % 2 == 0:
            x = self.se_block(x)
        residual = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.drop_path(x * self.layer_scale) + residual
        return x


class StageLayer(nn.Layer):
    def __init__(self, channels, depth, drop_rate, drop_path_list):
        super().__init__()
        self.blocks = nn.LayerList([
            RepViTBlock(channels, i, drop_rate, drop_path_list[i])
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Head(nn.Layer):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        # self.mid_linear = nn.Linear(dim, 1024)
        # self.act = nn.ReLU()
        # self.drop = nn.Dropout(0.2)
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = paddle.squeeze(x, (2, 3))
        # x = self.mid_linear(x)
        # x = self.act(x)
        # x = self.drop(x)
        x = self.linear(x)
        return x


class RepViT(nn.Layer):
    """
    CVPR 2024
    效果一般（可能是实现上有问题）
    """
    def __init__(self,
                 in_channels=3,
                 num_classes=19,
                 stage_channels=[64, 128, 256, 512],
                 stage_depths=[3, 3, 21, 3],
                 drop_rate=0.2,
                 drop_path_rate=0.2):
        super().__init__()
        self.stages = len(stage_channels)
        assert len(stage_depths) == self.stages
        dpr = [d for d in paddle.linspace(0, drop_path_rate, sum(stage_depths))]
        self.stem = Stem(in_channels, out_channels=stage_channels[0])
        self.layers = nn.LayerList([
            StageLayer(stage_channels[i],
                       stage_depths[i],
                       drop_rate,
                       dpr[sum(stage_depths[:i]):sum(stage_depths[:i]) + stage_depths[i]])
            for i in range(self.stages)
        ])
        self.down_layers = nn.LayerList([
            DownSampling(in_channels=stage_channels[i],
                         out_channels=stage_channels[i + 1])
            if i != self.stages - 1 else nn.Identity()
            for i in range(self.stages)
        ])
        self.head = Head(stage_channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for i in range(self.stages):
            x = self.layers[i](x)
            x = self.down_layers[i](x)
        x = self.head(x)
        return x
