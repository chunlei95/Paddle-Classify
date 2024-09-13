from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.vision_transformer import DropPath, trunc_normal_, zeros_, ones_


# class SwishImplementation(paddle.autograd.grad):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * F.sigmoid(i)
#         ctx.save_for_backward(i)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_tensors[0]
#         sigmoid_i = F.sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
#
#
# class MemoryEfficientSwish(nn.Layer):
#     def forward(self, x):
#         return SwishImplementation.apply(x)


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = paddle.stack([-x2, x1], axis=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2d(nn.Layer):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2D(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: paddle.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.transpose((0, 3, 1, 2))  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.transpose((0, 2, 3, 1))  # (b h w c)
        return x


class RetNetRelPos2d(nn.Layer):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** paddle.linspace(0, 1, embed_dim // num_heads // 2))
        # angle = angle.unsqueeze(-1).tile(1, 2).flatten()
        angle = angle.unsqueeze(-1)
        self.angle = paddle.tile(angle, (1, 2)).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        self.decay = paddle.log(
            1 - 2 ** (-initial_value - heads_range * paddle.arange(num_heads, dtype=paddle.float32) / num_heads))
        # self.register_buffer('angle', angle)
        # self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = paddle.arange(H).to(self.decay)
        index_w = paddle.arange(W).to(self.decay)
        grid = paddle.meshgrid([index_h, index_w])
        grid = paddle.stack(grid, axis=-1).reshape((H * W, 2))  # (H*W 2)
        mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        mask = (mask.abs()).sum(axis=-1)
        mask = mask * self.decay[:, None, None]  # (n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = paddle.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = paddle.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = paddle.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = paddle.arange(slen[0] * slen[1]).to(self.decay)
            sin = paddle.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape((slen[0], slen[1], -1))  # (h w d1)
            cos = paddle.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape((slen[0], slen[1], -1))  # (h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = paddle.arange(slen[0] * slen[1]).to(self.decay)
            sin = paddle.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape((slen[0], slen[1], -1))  # (h w d1)
            cos = paddle.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape((slen[0], slen[1], -1))  # (h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


class VisionRetentionChunk(nn.Layer):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias_attr=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias_attr=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierNormal()(self.q_proj.weight)
        nn.initializer.XavierNormal()(self.k_proj.weight)
        nn.initializer.XavierNormal()(self.v_proj.weight)
        nn.initializer.XavierNormal()(self.out_proj.weight)
        nn.initializer.XavierNormal()(self.out_proj.bias)

    def forward(self, x: paddle.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.shape

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.reshape((bsz, h, w, self.num_heads, self.key_dim)).transpose((0, 3, 1, 2, 4))  # (b n h w d1)
        k = k.reshape((bsz, h, w, self.num_heads, self.key_dim)).transpose((0, 3, 1, 2, 4))  # (b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''

        qr_w = qr.transpose((0, 2, 1, 3, 4))  # (b h n w d1)
        kr_w = kr.transpose((0, 2, 1, 3, 4))  # (b h n w d1)
        v = v.reshape((bsz, h, w, self.num_heads, -1)).transpose((0, 1, 3, 2, 4))  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose((0, 1, 2, 4, 3))  # (b h n w w)
        qk_mat_w = qk_mat_w + mask_w  # (b h n w w)
        qk_mat_w = F.softmax(qk_mat_w, -1)  # (b h n w w)
        v = paddle.matmul(qk_mat_w, v)  # (b h n w d2)

        qr_h = qr.transpose((0, 3, 1, 2, 4))  # (b w n h d1)
        kr_h = kr.transpose((0, 3, 1, 2, 4))  # (b w n h d1)
        v = v.transpose((0, 3, 2, 1, 4))  # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose((0, 1, 2, 4, 3))  # (b w n h h)
        qk_mat_h = qk_mat_h + mask_h  # (b w n h h)
        qk_mat_h = F.softmax(qk_mat_h, -1)  # (b w n h h)
        output = paddle.matmul(qk_mat_h, v)  # (b w n h d2)

        output = output.transpose((0, 3, 1, 2, 4)).flatten(3)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output


class VisionRetentionAll(nn.Layer):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias_attr=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias_attr=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierNormal()(self.q_proj.weight)
        nn.initializer.XavierNormal()(self.k_proj.weight)
        nn.initializer.XavierNormal()(self.v_proj.weight)
        nn.initializer.XavierNormal()(self.out_proj.weight)
        nn.initializer.Constant(0.0)(self.out_proj.bias)

    def forward(self, x: paddle.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.shape
        (sin, cos), mask = rel_pos

        assert h * w == mask.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.reshape((bsz, h, w, self.num_heads, -1)).transpose((0, 3, 1, 2, 4))  # (b n h w d1)
        k = k.reshape((bsz, h, w, self.num_heads, -1)).transpose((0, 3, 1, 2, 4))  # (b n h w d1)
        qr = theta_shift(q, sin, cos)  # (b n h w d1)
        kr = theta_shift(k, sin, cos)  # (b n h w d1)

        qr = qr.flatten(2, 3)  # (b n l d1)
        kr = kr.flatten(2, 3)  # (b n l d1)
        vr = v.reshape((bsz, h, w, self.num_heads, -1)).transpose((0, 3, 1, 2, 4))  # (b n h w d2)
        vr = vr.flatten(2, 3)  # (b n l d2)
        qk_mat = qr @ kr.transpose((0, 1, 3, 2))  # (b n l l)
        qk_mat = qk_mat + mask  # (b n l l)
        qk_mat = F.softmax(qk_mat, -1)  # (b n l l)
        output = paddle.matmul(qk_mat, vr)  # (b n l d2)
        output = output.transpose((0, 1, 3, 2)).reshape((bsz, h, w, -1))  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output


class FeedForwardNetwork(nn.Layer):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn=F.gelu,
            dropout=0.0,
            activation_dropout=0.0,
            layernorm_eps=1e-6,
            subln=False,
            subconv=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.dropout_module = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, epsilon=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: paddle.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RetBlock(nn.Layer):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False,
                 layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, epsilon=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, epsilon=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = self.create_parameter([1, 1, 1, embed_dim],
                                                 default_initializer=nn.initializer.Constant(layer_init_values))
            self.gamma_2 = self.create_parameter([1, 1, 1, embed_dim],
                                                 default_initializer=nn.initializer.Constant(layer_init_values))

    def forward(
            self,
            x: paddle.Tensor,
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
    ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent,
                                              incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x


class PatchMerging(nn.Layer):

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2D(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2D(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.transpose((0, 3, 1, 2))  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.transpose((0, 2, 3, 1))  # (b oh ow oc)
        return x


class BasicLayer(nn.Layer):

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96, drop_path: list | float = 0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging = None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.LayerList([
            RetBlock(flag, embed_dim, num_heads, ffn_dim,
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.shape
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        for blk in self.blocks:
            x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
                    retention_rel_pos=rel_pos)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class LayerNorm2d(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, epsilon=1e-6)

    def forward(self, x: paddle.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.transpose((0, 2, 3, 1))  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.transpose((0, 3, 1, 2))
        return x


class PatchEmbed(nn.Layer):

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2D(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2D(embed_dim // 2),
            nn.GELU(),
            nn.Conv2D(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2D(embed_dim // 2),
            nn.GELU(),
            nn.Conv2D(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2D(embed_dim),
            nn.GELU(),
            nn.Conv2D(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2D(embed_dim)
        )

    def forward(self, x):
        x = self.proj(x).transpose((0, 2, 3, 1))  # (b h w c)
        return x


class VisRetNet(nn.Layer):

    """
    CVPR 2024 将源代码转换为paddlepaddle版本，非自行复现
    学习率 0.0001左右，高了损失不降，低了接近于损失不降
    """

    def __init__(self, in_channels=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoints=[False, False, False, False],
                 chunkwise_recurrents=[True, True, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_channels, embed_dim=embed_dims[0],
                                      norm_layer=norm_layer if self.patch_norm else None)

        # stochastic depth
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.proj = nn.Linear(self.num_features, projection)
        self.norm = nn.BatchNorm2D(projection)
        # self.swish = MemoryEfficientSwish()
        self.swish = nn.Swish()
        self.avgpool = nn.AdaptiveAvgPool1D(1)
        self.head = nn.Linear(projection, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            try:
                zeros_(m.bias)
                ones_(m.weight)
            except:
                pass

    # @paddle.jit.ignore_module
    # def no_weight_decay(self):
    #     return {'absolute_pos_embed'}
    #
    # @paddle.jit.ignore_module
    # def no_weight_decay_keywords(self):
    #     return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x)  # (b h w c)
        x = self.norm(x.transpose((0, 3, 1, 2))).flatten(2, 3)  # (b c h*w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = paddle.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def RMT_T3(**kwargs):
    model = VisRetNet(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False],
        **kwargs
    )
    return model


def RMT_S(**kwargs):
    model = VisRetNet(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False],
        **kwargs
    )
    return model


def RMT_M2(**kwargs):
    model = VisRetNet(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[5, 5, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.4,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs
    )
    return model


def RMT_L6(**kwargs):
    model = VisRetNet(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        init_values=[2, 2, 2, 2],
        heads_ranges=[6, 6, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.5,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs
    )
    return model
