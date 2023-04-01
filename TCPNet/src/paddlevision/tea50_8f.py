import paddle.nn as nn
import math
import paddle
import paddle.nn.functional as F

from TCP.TCP_module import TCP


__all__ = ['Res2Net', 'res2net50']


class MEModule(nn.Layer):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=16, n_segment=8):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2D(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(num_features=self.channel//self.reduction)

        self.conv2 = nn.Conv2D(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias_attr=False)

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.sigmoid = nn.Sigmoid()

        # self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad = (0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2D(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias_attr=False)
        self.bn3 = nn.BatchNorm2D(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.shape
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w
        # t feature
        reshape_bottleneck = bottleneck.reshape((-1, self.n_segment) + tuple(bottleneck.shape[1:]))  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], axis=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.reshape((-1, self.n_segment) + tuple(conv_bottleneck.shape[1:]))
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment-1], axis=1)  # n, t-1, c//r, h, w
        
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
        # pad = (0,0,0,0,0,0,0,1)
        
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0, data_format='NDHWC')  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.reshape((-1,) + tuple(diff_fea_pluszero.shape[2:]))  #nt, c//r, h, w
        y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
        y = self.conv3(y)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        y = y - 0.5
        output = x + x * y.expand_as(x)
        return output

class ShiftModule(nn.Layer):
    """1Conv1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """

    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1D(
            2*self.fold, 2*self.fold,
            kernel_size=3, padding=1, groups=2, 
            bias_attr=False)
        # weight_size: (2*self.fold, 1, 3)
        if mode == 'shift':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = True
            with paddle.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[:self.fold, 0, 2] = 1 # shift left
                self.conv.weight[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
                if 2*self.fold < self.input_channels:
                    self.conv.weight[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            with paddle.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        # shift by conv
        # import pdb; pdb.set_trace()
        nt, c, h, w = x.shape
        n_batch = nt // self.n_segment
        x = x.reshape((n_batch, self.n_segment, c, h, w))
        x = x.transpose([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x = x.reshape((n_batch*h*w, c, self.n_segment))
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.reshape((n_batch, h, w, c, self.n_segment))
        x = x.transpose([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x = x.reshape((nt, c, h, w))
        return x

class Bottle2neckShift(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neckShift, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))

        self.me = MEModule(width*scale, reduction=16, n_segment=8)

        self.conv1 = nn.Conv2D(inplanes, width*scale, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width*scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2D(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        shifts = []
        for i in range(self.nums):
            convs.append(nn.Conv2D(width, width, kernel_size=3, stride=stride,
                padding=1, bias_attr=False))
            bns.append(nn.BatchNorm2D(width))
            shifts.append(ShiftModule(width, n_segment=8, n_div=2, mode='fixed'))
        shifts.append(ShiftModule(width, n_segment=8, n_div=2, mode='shift'))

        self.convs = nn.LayerList(convs)
        self.bns = nn.LayerList(bns)
        self.shifts = nn.LayerList(shifts)

        self.conv3 = nn.Conv2D(width*scale, planes * self.expansion,
                   kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        # import pdb; pdb.set_trace()
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.me(out)

        spx = paddle.split(out, self.width, 1)  # 4*(nt, c/4, h, w)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.shifts[i](sp)
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = paddle.cat((out, sp), 1)
        last_sp = spx[self.nums]
        last_sp = self.shifts[self.nums](last_sp)
        if self.scale != 1 and self.stype == 'normal':
            out = paddle.cat((out, last_sp), 1)
        elif self.scale != 1 and self.stype == 'stage':
            if self.stype =='stage' and spx[-1].shape[1] == 208:
                out = paddle.cat((out, last_sp), 1)
                # print(out.shape)
            else:
                out = paddle.cat((out, self.pool(last_sp)), 1)


        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottle2neck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2D(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2D(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2D(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2D(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2D(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2D(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        import pdb; pdb.set_trace()
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = paddle.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = paddle.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = paddle.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = paddle.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Layer):

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000,
                 TCP_module=None, segment=None,
                 ):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.num_segments = segment

        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if TCP_module is not None:
            print('Adding TCP module...')
            self.TCP = TCP_module
        else:
            self.avgpool = nn.AdaptiveAvgPool2D((1,1))
            self.TCP = None


        for m in self.named_parameters():
            if m == self.TCP :#reverse the initialization in TCP
                break
            elif isinstance(m, nn.Conv2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2D):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
            stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.TCP is not None:
            x = self.TCP(x)
        else :
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)

    return model

def tea50_8f(TCP_module=None, **kwargs):
    """Constructs a TEA model.
    part of the TEA model refers to the Res2Net-50_26w_4s.
    Args:
        TCP_module: if not None, generating TCP Net.
    """

    model = Res2Net(Bottle2neckShift, [3, 4, 6, 3], baseWidth = 26, scale = 4,
                    TCP_module=TCP_module, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']),
    #             strict=False)
    return model

def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)

    return model

def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth = 26, scale = 4, **kwargs)

    return model

def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 6, **kwargs)

    return model

def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 8, **kwargs)

    return model

def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 48, scale = 2, **kwargs)

    return model

def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 14, scale = 8, **kwargs)

    return model



if __name__ == '__main__':
    images = paddle.rand(8, 3, 224, 224)
    model = res2net50_14w_8s(pretrained=True)
    output = model(images)
    print(output.size())