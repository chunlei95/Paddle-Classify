from functools import partial

import paddle
import paddle.nn as nn

from models.van import VAN

if __name__ == '__main__':
    model_path = 'D:/PycharmProjects/Paddle-Classify/params/van-b2-crops.pdparams'
    model = VAN(class_num=19,
                drop_path_rate=0.2,
                drop_rate=0.2,
                embed_dims=[64, 128, 320, 512],
                mlp_ratios=[8, 8, 4, 4],
                norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                depths=[3, 3, 12, 3])
    model_params = paddle.load(model_path)
    model.set_state_dict(model_params)
    model.eval()
    input_spec = paddle.static.InputSpec([None, 3, 224, 224], dtype='float32', name='image')
    paddle.onnx.export(model,
                       'onnx_model',
                       input_spec=[input_spec])
