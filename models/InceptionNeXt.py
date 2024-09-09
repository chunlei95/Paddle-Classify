import functools

import numpy as np
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


class MlpHead(nn.Layer):
    def __init__(self, in_ch, mlp_ratio, num_classes, drop_rate):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, in_ch * mlp_ratio)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_ch * mlp_ratio, num_classes)
        self.norm = functools.partial(nn.LayerNorm, epsilon=1e-6)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = paddle.mean(x, axis=(2, 3))
        x = self.linear_1(x)
        x = self.act(x)
        # x = self.norm(x)
        x = self.drop(x)
        x = self.linear_2(x)
        return x


class ConvBNAct(nn.Layer):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, act_type=None):
        super().__init__()
        self.norm = nn.BatchNorm2D(out_ch)
        self.act = None
        if act_type is not None:
            self.act = act_type()
        self.conv = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, in_ch, ratio, drop_rate):
        super().__init__()
        self.conv_1 = nn.Conv2D(in_ch, in_ch * ratio, 1)
        self.act = nn.ReLU6()
        self.conv_2 = nn.Conv2D(in_ch * ratio, in_ch, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        residual = x
        x = self.conv_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv_2(x)
        x = self.drop(x)
        x = x + residual
        return x


class InceptionNeXtBlock(nn.Layer):
    def __init__(self, in_ch, mlp_ratio, group_ratio, kernel_size, drop_path_rate, drop_rate):
        super().__init__()
        group_ch = int(in_ch * group_ratio)
        self.conv_b1 = nn.Conv2D(group_ch, group_ch, 3, padding=1, groups=group_ch)
        self.conv_b2 = nn.Conv2D(group_ch, group_ch, (1, kernel_size), padding=(0, kernel_size // 2), groups=group_ch)
        self.conv_b3 = nn.Conv2D(group_ch, group_ch, (kernel_size, 1), padding=(kernel_size // 2, 0), groups=group_ch)
        self.norm = nn.BatchNorm2D(in_ch)
        self.mlp = Mlp(in_ch, mlp_ratio, drop_rate)
        self.drop_path = DropPath(drop_path_rate)
        self.split_indexes = (group_ch, group_ch, group_ch, in_ch - group_ch * 3)
        layer_scale_init_value = 1e-6
        self.layer_scale = self.create_parameter(shape=[in_ch, 1, 1],
                                                 default_initializer=Constant(value=layer_scale_init_value))

    def forward(self, x):
        residual = x
        x_hw, x_h, x_w, x_id = paddle.split(x, self.split_indexes, axis=1)
        x_hw = self.conv_b1(x_hw)
        x_h = self.conv_b2(x_h)
        x_w = self.conv_b3(x_w)
        x = paddle.concat([x_hw, x_h, x_w, x_id], axis=1)
        x = self.norm(x)
        x = self.mlp(x)
        x = residual + self.drop_path(x * self.layer_scale)
        return x


class BasicLayer(nn.Layer):
    def __init__(self, in_ch, depth, mlp_ratio, group_ratio, kernel_size, drop_path_list, drop_rate):
        super().__init__()
        self.blocks = nn.LayerList([
            InceptionNeXtBlock(in_ch, mlp_ratio, group_ratio, kernel_size, drop_path_list[i], drop_rate)
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DownSampling(nn.Layer):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.down = ConvBNAct(in_ch, out_ch, kernel_size, stride, 0)

    def forward(self, x):
        return self.down(x)


class InceptionNeXt(nn.Layer):
    """
    CVPR2024 https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_InceptionNeXt_When_Inception_Meets_ConvNeXt_CVPR_2024_paper.pdf
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 stage_channels,
                 stage_depths,
                 stage_group_ratios,
                 down_sample_kernels,
                 down_sample_strides,
                 stage_mlp_ratios,
                 head_ratio,
                 drop_path_rate,
                 drop_rate,
                 stage_kernel_sizes):
        super().__init__()
        self.stages = len(stage_channels)
        assert len(stage_depths) == self.stages == len(stage_mlp_ratios) == len(stage_group_ratios) == len(
            stage_kernel_sizes) == len(down_sample_kernels) == len(down_sample_strides)
        dpr = [d for d in np.linspace(0, drop_path_rate, sum(stage_depths))]
        stage_in_chs = [in_channels] + stage_channels[:-1]
        self.down_samples = nn.LayerList(
            [DownSampling(stage_in_chs[i],
                          stage_channels[i],
                          down_sample_kernels[i],
                          down_sample_strides[i]) for i in range(self.stages)]
        )
        self.stage_layers = nn.LayerList(
            [BasicLayer(in_ch=stage_channels[i],
                        depth=stage_depths[i],
                        mlp_ratio=stage_mlp_ratios[i],
                        group_ratio=stage_group_ratios[i],
                        kernel_size=stage_kernel_sizes[i],
                        drop_path_list=dpr[sum(stage_depths[:i]): sum(stage_depths[:i]) + stage_depths[i]],
                        drop_rate=drop_rate) for i in range(self.stages)]
        )
        self.head = MlpHead(in_ch=stage_channels[-1],
                            num_classes=num_classes,
                            mlp_ratio=head_ratio,
                            drop_rate=drop_rate)

    def forward(self, x):
        for i in range(self.stages):
            x = self.down_samples[i](x)
            x = self.stage_layers[i](x)
        x = self.head(x)
        return x


def InceptionNeXt_A(in_channels, num_classes):
    return InceptionNeXt(in_channels,
                         num_classes,
                         stage_channels=[40, 90, 180, 320],
                         stage_depths=[2, 2, 6, 2],
                         stage_group_ratios=[1 / 4, 1 / 4, 1 / 4, 1 / 4],
                         down_sample_kernels=[4, 4, 4, 4],
                         down_sample_strides=[2, 2, 2, 2],
                         stage_mlp_ratios=[4, 4, 4, 3],
                         head_ratio=3,
                         drop_path_rate=0.2,
                         drop_rate=0.1,
                         stage_kernel_sizes=[9, 9, 9, 9])


def InceptionNeXt_T(in_channels, num_classes):
    return InceptionNeXt(in_channels,
                         num_classes,
                         stage_channels=[64, 128, 320, 512],
                         stage_depths=[3, 3, 9, 3],
                         stage_group_ratios=[1 / 8, 1 / 8, 1 / 8, 1 / 8],
                         down_sample_kernels=[4, 4, 4, 4],
                         down_sample_strides=[2, 2, 2, 2],
                         stage_mlp_ratios=[4, 4, 4, 3],
                         head_ratio=3,
                         drop_path_rate=0.2,
                         drop_rate=0.1,
                         stage_kernel_sizes=[11, 11, 11, 11])


def InceptionNeXt_S(in_channels, num_classes):
    return InceptionNeXt(in_channels,
                         num_classes,
                         stage_channels=[96, 192, 384, 768],
                         stage_depths=[3, 3, 27, 3],
                         stage_group_ratios=[1 / 8, 1 / 8, 1 / 8, 1 / 8],
                         down_sample_kernels=[4, 4, 4, 4],
                         down_sample_strides=[2, 2, 2, 2],
                         stage_mlp_ratios=[4, 4, 4, 3],
                         head_ratio=3,
                         drop_path_rate=0.2,
                         drop_rate=0.1,
                         stage_kernel_sizes=[11, 11, 11, 11])


def InceptionNeXt_B(in_channels, num_classes):
    return InceptionNeXt(in_channels,
                         num_classes,
                         stage_channels=[128, 256, 512, 1024],
                         stage_depths=[3, 3, 27, 3],
                         stage_group_ratios=[1 / 8, 1 / 8, 1 / 8, 1 / 8],
                         down_sample_kernels=[4, 4, 4, 4],
                         down_sample_strides=[2, 2, 2, 2],
                         stage_mlp_ratios=[4, 4, 4, 3],
                         head_ratio=3,
                         drop_path_rate=0.2,
                         drop_rate=0.1,
                         stage_kernel_sizes=[11, 11, 11, 11])
