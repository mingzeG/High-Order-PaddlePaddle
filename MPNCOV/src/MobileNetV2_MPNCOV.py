from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
from representation import MPNCOV

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2D(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias_attr=False, weight_attr=ParamAttr(
                        initializer=KaimingNormal()),),
            nn.BatchNorm2D(out_planes),
            nn.ReLU6()
        )

class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False,  weight_attr=ParamAttr(
                        initializer=KaimingNormal()),),
            nn.BatchNorm2D(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_MPNCOV(nn.Layer):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2_MPNCOV, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.dimension_reduction = nn.Sequential(
               nn.Conv2D(self.last_channel, 256 , kernel_size=1, stride=1,padding = 0, bias_attr=False, weight_attr=ParamAttr(
                        initializer=KaimingNormal()),),
               nn.BatchNorm2D(256),
               nn.ReLU6()
             )

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32896 , num_classes),
        )

        # weight initialization
        print(dir(self))
        for m in self.state_dict():
            print(m)
            assert 0
            if isinstance(m, nn.Conv2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.dimension_reduction(x)
        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2_mpncov(pretrained=False, **kwargs):
    return MobileNetV2_MPNCOV(**kwargs)

if __name__ =='__main__':
    x = paddle.ones(shape=(2,3,224,224))
    model = MobileNetV2_MPNCOV()
    y = model(x)
    print(y.shape)