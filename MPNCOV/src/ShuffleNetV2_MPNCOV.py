from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn

from representation import MPNCOV

__all__ = [
    'ShuffleNetV2_MPNCOV', 'shufflenet_v2_mpncov_x0_5', 'shufflenet_v2_mpncov_x1_0',
    'shufflenet_v2_mpncov_x1_5', 'shufflenet_v2_mpncov_x2_0'
]

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = x.reshape([batchsize, groups, channels_per_group, height, width])

    x = x.transpose([0, 2, 1, 3, 4])

    # flatten
    x = x.reshape([batchsize, -1, height, width])

    return x


class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2D(inp),
                nn.Conv2D(inp, branch_features, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(branch_features),
                nn.ReLU(),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2D(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.BatchNorm2D(branch_features),
            nn.ReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2D(branch_features),
            nn.Conv2D(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.BatchNorm2D(branch_features),
            nn.ReLU(),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias_attr=False):
        return nn.Conv2D(i, o, kernel_size, stride, padding, bias_attr=bias_attr, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, axis=1)
            out = paddle.concat((x1, self.branch2(x2)), axis=1)
        else:
            out = paddle.concat((self.branch1(x), self.branch2(x)), axis=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2_MPNCOV(nn.Layer):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2_MPNCOV, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2D(input_channels, output_channels, 3, 2, 1, bias_attr=False),
            nn.BatchNorm2D(output_channels),
            nn.ReLU(),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2D(input_channels, output_channels, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(output_channels),
            nn.ReLU(),
        )
        self.dimension_reduction = nn.Sequential(
            nn.Conv2D(self._stage_out_channels[-1], 256 , kernel_size=1, stride=1,padding = 0, bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU6()
        )

        self.fc = nn.Linear(32896, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.dimension_reduction(x)
        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


def _shufflenetv2_mpncov(arch, pretrained=False,  *args, **kwargs):
    return ShuffleNetV2_MPNCOV( *args, **kwargs)


def shufflenet_v2_mpncov_x0_5(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2_mpncov('shufflenetv2_mpncov_x0.5', pretrained, 
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_mpncov_x1_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2_mpncov('shufflenetv2_mpncov_x1.0', pretrained, 
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_mpncov_x1_5(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2_mpncov('shufflenetv2_mpncov_x1.5', pretrained, 
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_mpncov_x2_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2_mpncov('shufflenetv2_mpncov_x2.0', pretrained, 
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)

if __name__ =='__main__':
    x = paddle.ones(shape=(2,3,224,224))
    model = shufflenet_v2_mpncov_x0_5()
    y = model(x)
    print(y.shape)