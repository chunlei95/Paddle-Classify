import paddle.nn as nn


class PatchEmbedding(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class DownSampling(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class AggregatedAttention(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ConvGLU(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class StageLayer(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Classifier(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class TransNeXt(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_classes,
                 stage_channels,
                 stage_depths,
                 drop_rate,
                 attn_drop,
                 drop_path_rate):
        super().__init__()

    def forward(self, x):
        pass
