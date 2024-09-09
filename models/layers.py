import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def to_2tuple(x):
    return tuple([x] * 2)


# noinspection PyProtectedMember,PyMethodMayBeStatic
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


class LKA(nn.Layer):
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


# noinspection PyProtectedMember,PyMethodMayBeStatic
class ConvNormAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm_type=None,
                 act_type=nn.GELU):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.norm = BuildNorm(out_channels, norm_type)
        self.act = act_type()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# noinspection PyProtectedMember,PyMethodMayBeStatic
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
        return self.conv_3(self.conv_2(self.conv_1(x)))


# noinspection PyProtectedMember,PyMethodMayBeStatic
class SegmentationHead(nn.Layer):
    def __init__(self, in_channels, num_classes, norm_type=None, act_type=nn.GELU):
        super().__init__()
        self.de_conv = nn.Conv2DTranspose(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1)
        self.norm = BuildNorm(in_channels, norm_type=norm_type)
        self.act = act_type()
        self.classifier = nn.Conv2D(in_channels, num_classes, kernel_size=3, padding=1)
        self.conv = ConvNormAct(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                padding=1,
                                norm_type=norm_type,
                                act_type=act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.de_conv(x, output_size=[x.shape[-2] * 2, x.shape[-1] * 2])
        x = self.norm(x)
        x = self.act(x)
        x = self.classifier(x)
        return x


class SegmentationHead_Semi(nn.Layer):
    def __init__(self, in_channels, norm_type=None, act_type=nn.GELU):
        super().__init__()
        self.de_conv = nn.Conv2DTranspose(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1)
        self.norm = BuildNorm(in_channels, norm_type=norm_type)
        self.act = act_type()
        self.conv_out = ConvNormAct(in_channels,
                                    in_channels,
                                    kernel_size=3,
                                    padding=1,
                                    norm_type=norm_type,
                                    act_type=act_type)
        self.conv = ConvNormAct(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                padding=1,
                                norm_type=norm_type,
                                act_type=act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.de_conv(x, output_size=[x.shape[-2] * 2, x.shape[-1] * 2])
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_out(x)
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
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


class GroupMlP(nn.Layer):
    def __init__(self, channels, mlp_drop, norm_type, act_type):
        super().__init__()
        self.norm_1 = BuildNorm(channels, norm_type)
        # 局部到全局的通道关系建模
        # 局部通道关系建模
        self.decompose = nn.Conv2D(channels, channels * 4, kernel_size=1, padding='same', groups=channels)
        self.norm_2 = BuildNorm(channels * 4, norm_type)
        self.combined = nn.Conv2D(channels * 4, channels, kernel_size=1, padding='same', groups=channels)
        self.act = act_type()
        # 全局通道关系建模
        self.ci = nn.Conv2D(channels, channels, kernel_size=1)
        self.drop = nn.Dropout2D(mlp_drop)

    def forward(self, x):
        x = self.norm_1(x)
        residual = x
        x = self.decompose(x)
        x = self.act(x)
        # x = self.norm_2(x)
        x = self.combined(x)
        x = self.ci(x)
        x = self.drop(x)
        x = x + residual
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class SkipLayer(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channel_update = nn.Conv2D(in_channels=channels * 2,
                                        out_channels=channels,
                                        kernel_size=1)

    def forward(self, x, skip_x):
        if x.shape[2] != skip_x.shape[2] or x.shape[3] != skip_x.shape[3]:
            x = F.interpolate(x, [skip_x.shape[2], skip_x.shape[3]])
        x = paddle.concat([x, skip_x], axis=1)
        x = self.channel_update(x)
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class MLP(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.,
                 norm_type=nn.LayerNorm,
                 act_type=nn.GELU):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = BuildNorm(in_channels, norm_type=norm_type)
        # self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.conv1 = nn.Conv2D(in_channels, hidden_channels, 3, 1, 1)
        self.act = act_type()
        self.conv2 = nn.Conv2D(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        # x1 = self.avg_pool(x)
        # x2 = self.max_pool(x)
        # x = x1 * x2
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class Mlp(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.,
                 norm_type=nn.LayerNorm,
                 act_type=nn.GELU):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = BuildNorm(in_channels, norm_type=norm_type)
        self.conv1 = nn.Conv2D(in_channels, hidden_channels, 1, 1)
        self.act = act_type()
        self.conv2 = nn.Conv2D(hidden_channels, out_channels, 1, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class LocalBlock(nn.Layer):
    def __init__(self, channels, kernel_size, norm_type, act_type):
        super().__init__()
        self.conv_1 = nn.Conv2D(channels, channels, kernel_size, 1, (kernel_size - 1) // 2, groups=channels)
        self.conv_2 = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)
        self.conv_3 = nn.Conv2D(channels, channels, 3, 1, 1, groups=channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class FFN(nn.Layer):
    def __init__(self, channels, norm_type, act_type):
        super().__init__()
        self.norm = BuildNorm(channels, norm_type=norm_type)
        self.act = act_type()
        self.conv1x1_1 = nn.Conv2D(channels, channels * 4, 1)
        self.conv1x1_2 = nn.Conv2D(channels * 4, channels, 1)
        self.conv3x1 = nn.Conv2D(channels * 4, channels * 4, 3, 1, 1, groups=channels * 4)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1x1_1(x)
        residual = x
        x = self.conv3x1(x)
        x = x + residual
        x = self.drop(x)
        x = self.act(x)
        x = self.conv1x1_2(x)
        x = self.drop(x)
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
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


# noinspection PyMethodMayBeStatic,PyProtectedMember
class PatchMerging(nn.Layer):
    """
    Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        # self.reduction = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Args:
            x: Input feature, tensor size (B, L, C).
            H, W: Spatial resolution of the input feature.
        """
        # B, L, C = x.shape
        # H = W = int(math.sqrt(L))
        B, C, H, W = x.shape

        # x = x.reshape([B, H, W, C])
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

        # x = paddle.transpose(x, (0, 2, 3, 1))

        x = self.norm(x)
        # x = paddle.transpose(x, (0, 3, 1, 2))
        x = self.reduction(x)
        x = paddle.transpose(x, (0, 3, 1, 2))
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
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


class PatchSplitSelectDown(nn.Layer):
    def __init__(self, channels, norm_type):
        super().__init__()
        self.conv_1 = nn.Conv2D(channels * 4, channels, 1, groups=channels)
        self.conv_2 = nn.Conv2D(channels, channels * 4, 1, groups=channels)
        self.act_1 = nn.GELU()
        self.act_2 = nn.Sigmoid()
        self.norm = BuildNorm(channels * 4, norm_type=norm_type)
        self.li = nn.Conv2D(channels * 4, channels * 2, 3, 1, 1, groups=channels * 2)
        self.ci = nn.Conv2D(channels * 2, channels * 2, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # x = paddle.transpose(x, (0, 2, 3, 1))
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # x = x.transpose([0, 3, 1, 2])
            x = F.pad(x, [0, W % 2, 0, H % 2])
            # x = x.transpose([0, 2, 3, 1])
        H_, W_ = x.shape[2:]
        # x = paddle.transpose(x, (0, 2, 3, 1))
        x0 = x[:, :, 0:H_ // 2, 0:W_ // 2]  # B H/2 W/2 C
        x1 = x[:, :, 0:H_ // 2, W_ // 2:]  # B H/2 W/2 C
        x2 = x[:, :, H_ // 2:, 0:W_ // 2]  # B H/2 W/2 C
        x3 = x[:, :, H_ // 2:, W_ // 2:]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], 1)
        x = self.norm(x)
        x_ = self.conv_1(x)
        x_ = self.act_1(x_)
        x_ = self.conv_2(x_)
        x_ = self.act_2(x_)
        x = x * x_
        x = self.li(x)
        x = self.ci(x)
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class ConvDownSample(nn.Layer):
    def __init__(self, in_channels, out_channels, norm_type=None):
        super().__init__()
        self.norm = BuildNorm(in_channels, norm_type=norm_type)
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class PoolDownSample(nn.Layer):
    def __init__(self, in_dim, out_dim, norm_type=None):
        super().__init__()
        self.norm = BuildNorm(in_dim, norm_type=norm_type)
        self.down = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.proj = nn.Conv2D(in_dim, out_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.down(x)
        x = self.proj(x)
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class PatchExpand(nn.Layer):
    def __init__(self,
                 dim,
                 dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Conv2D(
            dim, self.dim_scale * dim, 1, bias_attr=False)
        self.norm = BuildNorm(dim, norm_type=norm_layer)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        # B, C, H, W -> B, H, W, C
        # x = paddle.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.expand(x)
        x = paddle.transpose(x, (0, 2, 3, 1))
        C = x.shape[-1]
        # B, L, C = x.shape
        # assert L == H * W, "input features has wrong size"

        # x = x.reshape((B, H, W, C))
        x = x.reshape((B, H, W, 2, 2, C // 4))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * 2, W * 2, C // 4))
        # x = x.reshape((B, H * 2 * W * 2, C // 4))

        # 输出形状为B, C, H, W
        x = paddle.transpose(x, (0, 3, 1, 2))

        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class PatchDecompose(nn.Layer):
    def __init__(self, dim, base_channels, norm_layer, dim_scale=2, reduce_dim=True):
        super().__init__()
        assert dim // dim_scale
        self.dim = dim
        new_dim = dim // dim_scale if reduce_dim else dim
        self.expand_scale = 4
        self.expand = nn.Conv2D(dim, self.expand_scale * dim, kernel_size=1, padding='same', groups=base_channels)
        self.compress = nn.Conv2D(dim, new_dim, kernel_size=1, padding='same', groups=base_channels)
        self.channel_interact = nn.Conv2D(new_dim, new_dim, 1)
        self.norm = BuildNorm(dim, norm_layer)

    def forward(self, x):
        B, C, H, W = x.shape
        # new_x = paddle.zeros([B, C, 2 * H, 2 * W])
        x = self.norm(x)
        x = self.expand(x)
        # new_x[:, :, 0::2, 0::2] = x[:, 0::self.expand_scale, :, :]
        # new_x[:, :, 0::2, 1::2] = x[:, 1::self.expand_scale, :, :]
        # new_x[:, :, 1::2, 0::2] = x[:, 2::self.expand_scale, :, :]
        # new_x[:, :, 1::2, 1::2] = x[:, 3::self.expand_scale, :, :]
        # x = new_x

        x = paddle.transpose(x, (0, 2, 3, 1))
        C = x.shape[-1]
        x = x.reshape((B, H, W, 2, 2, C // 4))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * 2, W * 2, C // 4))
        # 输出形状为B, C, H, W
        x = paddle.transpose(x, (0, 3, 1, 2))
        x = self.compress(x)
        x = self.channel_interact(x)
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class FinalPatchExpandX4(nn.Layer):
    def __init__(self,
                 dim,
                 dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 4 * dim, bias_attr=False)
        self.output_dim = dim
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, H, W = x.shape
        # B, C, H, W -> B, H, W, C
        x = paddle.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.expand(x)
        C = x.shape[-1]
        # B, L, C = paddle.shape(x)[0:3]
        # assert L == H * W, "input features has wrong size"

        # x = x.reshape((B, H, W, C))
        x = x.reshape((B, H, W, self.dim_scale, self.dim_scale,
                       C // (self.dim_scale ** 2)))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * self.dim_scale, W * self.dim_scale,
                       C // (self.dim_scale ** 2)))
        # x = x.reshape((B, -1, self.output_dim))

        # 输出形状为B, C, H, W
        x = paddle.transpose(x, (0, 3, 1, 2))
        return x


# noinspection PyMethodMayBeStatic,PyProtectedMember
class FinalPatchExpandX2(nn.Layer):
    def __init__(self,
                 dim,
                 dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 4 * dim, bias_attr=False)
        self.output_dim = dim
        self.norm = BuildNorm(dim, norm_type=norm_layer)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, H, W = x.shape
        # B, C, H, W -> B, H, W, C
        # x = paddle.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = paddle.transpose(x, (0, 2, 3, 1))
        x = self.expand(x)
        C = x.shape[-1]
        # B, L, C = paddle.shape(x)[0:3]
        # assert L == H * W, "input features has wrong size"

        # x = x.reshape((B, H, W, C))
        x = x.reshape((B, H, W, self.dim_scale, self.dim_scale,
                       C // (self.dim_scale ** 2)))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * self.dim_scale, W * self.dim_scale,
                       C // (self.dim_scale ** 2)))
        # x = x.reshape((B, -1, self.output_dim))

        # 输出形状为B, C, H, W
        x = paddle.transpose(x, (0, 3, 1, 2))
        return x


class SkipWithChannelAlign(nn.Layer):
    """Skip layer with channel align"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.channel_fix = nn.Conv2D(in_channels=channels * 2,
                                     out_channels=channels,
                                     kernel_size=1,
                                     groups=channels)
        self.ci = nn.Conv2D(channels, channels, 1)

    def forward(self, x, skip_x):
        x = paddle.concat([x, skip_x], axis=1)
        new_x = paddle.zeros_like(x)
        new_x[:, 0::2, :, :] = x[:, 0:self.channels, :, :]
        new_x[:, 1::2, :, :] = x[:, self.channels:, :, :]
        x = self.channel_fix(new_x)
        x = self.ci(x)
        return x


class ModulationSkipLayer(nn.Layer):
    def __init__(self, in_channels, out_channels=None, norm_type=nn.LayerNorm, drop_rate=0.5):
        super().__init__()
        out_channels = out_channels or in_channels
        self.proj_1 = nn.Sequential(
            nn.Conv2D(in_channels * 2, out_channels, 1),
            BuildNorm(out_channels, norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1),
        )
        self.residual_proj = nn.Sequential(
            nn.Conv2D(in_channels * 2, out_channels, 1),
        )
        self.local = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1, 1)
        )
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, skip_x):
        x = paddle.concat([x, skip_x], axis=1)
        residual = x
        x = self.proj_1(x)
        mod = self.local(x)
        mod = self.dropout(mod)
        x = x * mod
        x = self.proj_2(x)
        residual = self.residual_proj(residual)
        x = x + residual
        return x


class ModulationSkipLayerV2(nn.Layer):
    def __init__(self, in_channels, base_channels, out_channels=None, norm_type=nn.LayerNorm, drop_rate=0.5):
        super().__init__()
        out_channels = out_channels or in_channels
        self.proj_1 = nn.Sequential(
            nn.Conv2D(in_channels * 2, out_channels, 1),
            BuildNorm(out_channels, norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1),
        )
        self.residual_proj = nn.Sequential(
            nn.Conv2D(in_channels * 2, out_channels, 1),
        )
        self.local = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1, 1)
        )
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, skip_x):
        x = paddle.concat([x, skip_x], axis=1)
        residual = x
        x = self.proj_1(x)
        mod = self.local(x)
        # sn = sum([sn, mod])
        # # sn = self.sn_proj(sn)
        # sn = self.act(sn)
        mod = self.dropout(mod)
        x = x * mod
        x = self.proj_2(x)
        residual = self.residual_proj(residual)
        x = x + residual
        return x


class ModulationSkipLayerV2_NoSN(nn.Layer):
    def __init__(self, in_channels, base_channels, out_channels=None, norm_type=nn.LayerNorm, drop_rate=0.5):
        super().__init__()
        out_channels = out_channels or in_channels
        self.proj_1 = nn.Sequential(
            nn.Conv2D(in_channels * 2, out_channels, 1),
            BuildNorm(out_channels, norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1),
        )
        self.residual_proj = nn.Sequential(
            nn.Conv2D(in_channels * 2, out_channels, 1),
        )
        self.local = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1, 1)
        )
        # self.sn_proj = nn.Sequential(
        #     nn.Conv2D(out_channels, out_channels, 3, 1, 'same', groups=out_channels),
        #     nn.Conv2D(out_channels, out_channels, 1)
        # )

        # self.sn_proj = nn.Sequential(
        #     nn.Conv2D(out_channels, out_channels, 3, 1, 'same', groups=base_channels),
        #     nn.Conv2D(out_channels, out_channels, 1)
        # )
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, skip_x):
        if x.shape[2:] != skip_x.shape[2:]:
            skip_x = F.interpolate(skip_x, x.shape[2:])
        x = paddle.concat([x, skip_x], axis=1)
        residual = x
        x = self.proj_1(x)
        mod = self.local(x)
        # sn = sum([sn, mod])
        # sn = self.sn_proj(sn)
        mod = self.dropout(mod)
        x = x * mod
        x = self.proj_2(x)
        residual = self.residual_proj(residual)
        x = x + residual
        return x


class ModulationSkipLayerV3(nn.Layer):
    def __init__(self, in_channels, out_channels=None, norm_type=nn.LayerNorm, drop_rate=0.5):
        super().__init__()
        out_channels = out_channels or in_channels
        self.proj_1 = nn.Sequential(
            nn.Conv2D(in_channels * 2, out_channels, 1),
            BuildNorm(out_channels, norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1),
        )
        self.residual_proj = nn.Sequential(
            nn.Conv2D(in_channels * 2, out_channels, 1),
        )
        self.local = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 3, 1, 'same', groups=out_channels),
            nn.Conv2D(out_channels, out_channels, 1, 1)
        )
        self.sn_proj = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 3, 1, 'same', groups=out_channels),
            nn.Conv2D(out_channels, out_channels, 1)
        )
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, skip_x, sn):
        x = paddle.concat([x, skip_x], axis=1)
        residual = x
        x = self.proj_1(x)
        mod = self.local(x)
        sn = sum([sn, mod])
        sn = self.sn_proj(sn)
        sn = self.dropout(sn)
        x = x * sn
        x = self.proj_2(x)
        residual = self.residual_proj(residual)
        x = x + residual
        return x, sn


class SkipLayerV2(nn.Layer):
    def __init__(self, in_channels, out_channels=None, norm_type=nn.LayerNorm, drop_rate=0.5):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm = BuildNorm(in_channels * 2, norm_type)
        self.e_proj = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, 3, 1, 'same', groups=in_channels),
            BuildNorm(in_channels, norm_type),
            nn.Conv2D(in_channels, in_channels, 1)
        )
        self.proj = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1),
            # BuildNorm(out_channels, norm_type)
        )
        self.proj_1 = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1),
            # BuildNorm(out_channels, norm_type)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1),
            # BuildNorm(out_channels, norm_type)
        )
        # self.residual_proj = nn.Sequential(
        #     nn.Conv2D(in_channels * 2, out_channels, 1),
        #     BuildNorm(out_channels, norm_type)
        # )
        # self.local = nn.Sequential(
        #     nn.Conv2D(out_channels, out_channels, 1, 1)
        #     # BuildNorm(out_channels, norm_type),
        #     # nn.Conv2D(out_channels, out_channels, 1),
        # )
        self.dropout = nn.Dropout2D(drop_rate)

    def forward(self, x, skip_x):
        residual = x
        residual = self.proj_2(residual)
        skip_x = self.e_proj(skip_x)
        x_mod = self.proj_1(x)
        x_mod = self.dropout(x_mod)
        x = x_mod * skip_x
        x = self.proj(x)
        x = x + residual
        return x


class NewSkipLayer(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2D(channels, channels, 1)
        self.proj_1 = nn.Conv2D(channels * 2, channels, 1)

    def forward(self, x, skip_x):
        s = self.conv(skip_x)
        x = paddle.concat([s, x], axis=1)
        x = self.proj_1(x)
        return x


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for layer in self._sub_layers.values():
            inputs = layer(*inputs)
        return inputs


class UNet_Decoder(nn.Layer):
    def __init__(self,
                 channels,
                 num_classes,
                 base_channels,
                 norm_type,
                 act_type,
                 basic_block,
                 **kwargs):
        super().__init__()
        self.patch_expand_1 = PatchDecompose(dim=channels[3],
                                             base_channels=base_channels,
                                             norm_layer=norm_type,
                                             dim_scale=2,
                                             reduce_dim=True)
        self.patch_expand_2 = PatchDecompose(dim=channels[2],
                                             base_channels=base_channels,
                                             norm_layer=norm_type,
                                             dim_scale=2,
                                             reduce_dim=True)
        self.patch_expand_3 = PatchDecompose(dim=channels[1],
                                             base_channels=base_channels,
                                             norm_layer=norm_type,
                                             dim_scale=2,
                                             reduce_dim=True)
        self.final_expand = PatchDecompose(dim=channels[0],
                                           base_channels=base_channels,
                                           norm_layer=norm_type,
                                           dim_scale=2,
                                           reduce_dim=False)

        self.skip_1 = SkipLayer(channels=channels[2])
        self.skip_2 = SkipLayer(channels=channels[1])
        self.skip_3 = SkipLayer(channels=channels[0])

        self.layer_1 = nn.Sequential(*[basic_block(in_channels=channels[2], **kwargs)] * 2)
        self.layer_2 = nn.Sequential(*[basic_block(in_channels=channels[1], **kwargs)] * 2)
        self.layer_3 = nn.Sequential(*[basic_block(in_channels=channels[0], **kwargs)] * 2)

        self.output_stem = SegmentationHead(in_channels=channels[0],
                                            num_classes=num_classes,
                                            norm_type=norm_type,
                                            act_type=act_type)

    def forward(self, x_encoder):
        x = x_encoder[-1]
        skip_x = x_encoder[:-1]
        x = self.patch_expand_1(x)
        x = self.skip_1(x, skip_x[2])
        x = self.layer_1(x)

        x = self.patch_expand_2(x)
        x = self.skip_2(x, skip_x[1])
        x = self.layer_2(x)

        x = self.patch_expand_3(x)
        x = self.skip_3(x, skip_x[0])
        x = self.layer_3(x)

        x = self.final_expand(x)
        x = self.output_stem(x)
        return x
