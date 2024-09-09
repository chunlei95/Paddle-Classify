import math

import paddle
import paddle.nn as nn

from models.layers import PatchCombined, OverlapPatchEmbed, \
    ConvStem, SegmentationHead, BuildNorm, ConditionalPositionEncoding, \
    SkipLayer, PatchDecompose


# noinspection PyProtectedMember,PyMethodMayBeStatic
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


# noinspection PyProtectedMember,PyMethodMayBeStatic,PyDefaultArgument,SpellCheckingInspection
class ConvAttn(nn.Layer):
    def __init__(self, in_channels, kernel_sizes=[7, 11, 21], dilations=[3, 3, 3], drop_rate=0.5):
        super().__init__()
        self.scale_layers = nn.LayerList([
            LKA(channels=in_channels, kernel_size=kernel_size, dilation=dilation) for kernel_size, dilation in
            zip(kernel_sizes, dilations)
        ])
        self.base_scale_dwconv3x3 = nn.Conv2D(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.scale_interactive = nn.Conv2D(in_channels, in_channels, 1)
        self.drop_out = nn.Dropout2D(drop_rate)

    def forward(self, x):
        x_base_scale = self.base_scale_dwconv3x3(x)
        x_other_scale = sum([scale_layer(x) for scale_layer in self.scale_layers])
        attn = x_base_scale + x_other_scale
        attn = self.scale_interactive(attn)
        attn = self.drop_out(attn)
        x = attn * x
        return x


# noinspection PyDefaultArgument,PyMethodMayBeStatic,PyProtectedMember
class BasicBlock(nn.Layer):
    def __init__(self, channels, encode_size, local_kernel_size, drop_rate, attn_kernel_size=[7, 11, 21],
                 dilations=[3, 3, 3],
                 conv_attn_num=1,
                 norm_type=nn.LayerNorm,
                 act_type=nn.GELU):
        super().__init__()
        self.cpe = ConditionalPositionEncoding(channels, encode_size=encode_size)
        self.norm = BuildNorm(channels, norm_type=norm_type)
        self.local = nn.Sequential(*[
            nn.Conv2D(in_channels=channels, out_channels=channels, kernel_size=local_kernel_size,
                      padding=(local_kernel_size - 1) // 2, groups=channels),
            BuildNorm(channels, norm_type),
            nn.Conv2D(in_channels=channels, out_channels=channels * 4, kernel_size=1),
            act_type(),
            nn.Conv2D(in_channels=channels * 4, out_channels=channels, kernel_size=1)
        ])
        self.conv_attn = nn.Sequential(*[ConvAttn(in_channels=channels, kernel_sizes=attn_kernel_size,
                                                  dilations=dilations,
                                                  drop_rate=drop_rate)] * conv_attn_num)
        self.proj = nn.Conv2D(in_channels=channels, out_channels=channels, kernel_size=1)
        # self.mlp = Mlp(in_channels=channels,
        #                hidden_channels=channels * 4,
        #                drop_rate=drop_rate,
        #                norm_type=norm_type)

    def forward(self, x):
        x = self.cpe(x)
        residual = x

        x = self.norm(x)
        x = self.local(x)
        x = self.conv_attn(x)
        x = self.proj(x)
        x = x + residual

        # residual = x
        # x = self.mlp(x)
        # x = x + residual
        return x


class BasicLayer(nn.Layer):
    def __init__(self, channels, depth, encode_size, drop_rate, local_kernel_size, attn_kernel_size, dilations,
                 conv_attn_num, norm_type, act_type):
        super().__init__()
        assert depth >= 1, 'depth must greater or equal than 1!'
        assert len(attn_kernel_size) == len(dilations)
        self.layers = [BasicBlock(
            channels=channels,
            drop_rate=drop_rate,
            encode_size=encode_size,
            local_kernel_size=local_kernel_size,
            attn_kernel_size=attn_kernel_size,
            dilations=dilations,
            conv_attn_num=conv_attn_num,
            norm_type=norm_type,
            act_type=act_type
        )] * depth
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        # B, C, H, W
        x = self.layers(x)
        # 输出形状为B, C, H, W
        return x


# noinspection PyDefaultArgument
# @manager.MODELS.add_component
class ConvAttnUNet(nn.Layer):

    # noinspection PyTypeChecker
    def __init__(self,
                 in_channels=3,
                 num_classes=2,
                 num_stages=4,
                 patch_size=3,
                 merge_size=3,
                 encode_size=3,
                 stage_out_channels=[64, 128, 256, 512],
                 depths=[2, 2, 2, 2],
                 local_kernel_sizes=[3, 3, 3, 3],
                 attn_kernel_sizes=[[7, 9, 11],
                                    [7, 9, 11],
                                    [7, 9, 11],
                                    [7, 9, 11]],
                 dilations=[[3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3]],
                 drop_rate=0.5,
                 conv_attn_num=1,
                 norm_type=None,
                 act_type=nn.GELU):
        super().__init__()
        assert len(attn_kernel_sizes) == len(dilations) == num_stages == len(local_kernel_sizes) == len(
            stage_out_channels)
        if norm_type is None:
            raise RuntimeWarning('norm type is not specified! there is no normalization in the model!')
        if type(norm_type) == str:
            norm_type = eval(norm_type)
        if type(act_type) == str:
            act_type = eval(act_type)
        self.conv_stem = ConvStem(in_channels=in_channels,
                                  out_channels=stage_out_channels[0],
                                  norm_type=norm_type,
                                  act_type=act_type)
        self.patch_embed = OverlapPatchEmbed(in_channels=stage_out_channels[0],
                                             out_channels=stage_out_channels[0],
                                             patch_size=patch_size,
                                             stride=2)

        self.stage_encoder_layers = nn.LayerList()
        self.stage_decoder_layers = nn.LayerList()
        self.stage_merge = nn.LayerList()
        self.stage_expand = nn.LayerList()
        self.skip_layers = nn.LayerList()
        base_channels = stage_out_channels[0]
        self.final_expand = PatchDecompose(dim=stage_out_channels[0],
                                           base_channels=base_channels,
                                           dim_scale=2,
                                           norm_layer=norm_type,
                                           reduce_dim=False
                                           )
        self.segmentation_head = SegmentationHead(in_channels=stage_out_channels[0],
                                                  num_classes=num_classes,
                                                  norm_type=norm_type,
                                                  act_type=act_type)
        patch_expands = []
        skip_connections = []
        decoder_depth = [1, 2, 2, 2]
        for i in range(num_stages):
            encoder_layers = BasicLayer(channels=stage_out_channels[i],
                                        depth=depths[i],
                                        encode_size=encode_size,
                                        drop_rate=drop_rate,
                                        local_kernel_size=local_kernel_sizes[i],
                                        attn_kernel_size=attn_kernel_sizes[i],
                                        dilations=dilations[i],
                                        conv_attn_num=conv_attn_num,
                                        norm_type=norm_type,
                                        act_type=act_type)
            decoder_layers = BasicLayer(channels=stage_out_channels[num_stages - 1 - i],
                                        depth=decoder_depth[i],
                                        encode_size=encode_size,
                                        drop_rate=drop_rate,
                                        local_kernel_size=local_kernel_sizes[num_stages - 1 - i],
                                        attn_kernel_size=attn_kernel_sizes[num_stages - 1 - i],
                                        dilations=dilations[i],
                                        conv_attn_num=conv_attn_num,
                                        norm_type=norm_type,
                                        act_type=act_type)

            merge = True if i != (num_stages - 1) else False
            merge_layer = PatchCombined(dim=stage_out_channels[i],
                                        merge_size=merge_size,
                                        norm_layer=norm_type) if merge else nn.Identity()
            self.stage_encoder_layers.append(encoder_layers)
            self.stage_decoder_layers.append(decoder_layers)
            self.stage_merge.append(merge_layer)

            expand = False if i == (num_stages - 1) else True
            patch_expands.append(PatchDecompose(dim=stage_out_channels[i + 1],
                                                base_channels=base_channels,
                                                dim_scale=2,
                                                norm_layer=nn.LayerNorm) if expand else nn.Identity())

            skip = False if i == (num_stages - 1) else True
            skip_connections.append(SkipLayer(channels=stage_out_channels[i]) if skip else nn.Identity())
        patch_expands.reverse()
        skip_connections.reverse()
        self.stage_expand.extend(patch_expands)
        self.skip_layers.extend(skip_connections)
        self.norm_encoder = BuildNorm(stage_out_channels[0], norm_type)
        self.act_encoder = act_type()
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.head = nn.Linear(stage_out_channels[0], num_classes)

    def forward(self, x):
        skip_features = []
        x = self.conv_stem(x)
        x = self.patch_embed(x)
        for encoder_layers, merge_layer in zip(self.stage_encoder_layers, self.stage_merge):
            x = encoder_layers(x)
            # x的形状为B, C, H, W
            skip_features.append(x)
            x = merge_layer(x)

        skip_features.reverse()

        for skip_x, decoder_layers, expand_layer, skip_layer in zip(skip_features, self.stage_decoder_layers,
                                                                    self.stage_expand,
                                                                    self.skip_layers):
            if type(expand_layer) is not nn.Identity:
                x = expand_layer(x)
            if type(skip_layer) is not nn.Identity:
                x = skip_layer(x, skip_x)
            x = decoder_layers(x)
        x = self.act_encoder(self.norm_encoder(x))
        x = self.pool(x)
        x = paddle.squeeze(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    net = ConvAttnUNet(norm_type=nn.LayerNorm)
    print(net)
    x = paddle.randn((2, 1, 224, 224))
    out = net(x)
    for o in out:
        print(o.shape)
    # print(out.shape)
