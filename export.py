from functools import partial

import paddle
import paddle.nn as nn

from models.van import VAN_B3

if __name__ == '__main__':
    model_path = 'D:/PycharmProjects/van_b3_crop_identity.pdparams'
    model = VAN_B3(class_num=19, img_size=256)
    model_params = paddle.load(model_path)
    model.set_state_dict(model_params)
    model.eval()
    input_spec = paddle.static.InputSpec([None, 3, 256, 256], dtype='float32', name='image')
    paddle.onnx.export(model,
                       'onnx_model',
                       input_spec=[input_spec])
