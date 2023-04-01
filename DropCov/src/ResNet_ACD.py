import paddle
import paddle.nn as nn
import numpy as np
from representation.covariance  import *
from representation.eca_layer import *

__all__ = ['ResNet_ACD', 'resnet18_ACD', 'resnet34_ACD', 'resnet50_ACD', 
           'resnet101_ACD','resnet152_ACD']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias_attr=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet_ACD(nn.Layer):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_ACD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

        self.eca_layer = eca_layer(channel=256)
        
        self.cov = Covariance()

        # triu 256
        self.layer_reduce = nn.Conv2D(512 * block.expansion,256,kernel_size = 1,stride = 1,padding = 0,bias_attr = False)
        self.layer_reduce_bn = nn.BatchNorm2D(256)
        self.layer_reduce_relu = nn.ReLU()
        self.fc = nn.Linear(32896, num_classes)

        
        for m in self.parameters():
            if isinstance(m, nn.Conv2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias_attr, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.parameters():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _triuvec(self, x):
         return Triuvec.apply(x)

    def _signed_sqrt(self, x):
        x = paddle.mul(x.sign(), paddle.sqrt(x.abs()+ 1e-8))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x
        
    # Keep larger entities 0.5 0.7    
    def cov_value_drop(self, x, p):
        if not self.training:
            return self._triuvec(self.cov(x))
        
        x = self._triuvec(self.cov(x))
        mask = paddle.zeros(size = x.shape).cuda()
        sort_sum, index = paddle.sort(x, dim=1, descending= True)
        mask_index = index[:, :int(x.shape[1] * p)]
        mask = mask.scatter_(1, mask_index, 1)
        x = x.mul(mask)
        return x   

    # Uniformly drop entities with interval of 1 or 2.   
    def cov_uniform_value_drop(self, x, interval = 1):
        if not self.training:
            return self._triuvec(self.cov(x))
        
        x = self._triuvec(self.cov(x))
        if interval == 1:
            random_idx = paddle.randint(0,2, (1,))
            x[:, random_idx::2] = 0
        elif interval == 2:
            random_idx = paddle.randint(0,3, (1,))
            x[:, random_idx::3] = 0
            x[:, random_idx + 1::3] = 0
        return x   

    # Uniformly drop entities with interval of 1 or 2 for ranking entities.   
    def cov_rank_value_uniform_drop(self, x, interval = 1):
        if not self.training:
            return self._triuvec(self.cov(x))
        
        x = self._triuvec(self.cov(x))
        mask = paddle.zeros(size = x.shape).cuda()
        sort_sum, index = paddle.sort(x, dim=1, descending= True)
        if interval == 1:
            mask_index = index[:, 1::2]
        elif interval == 2:
            mask_index = index[:, 1::3]
        mask = mask.scatter_(1, mask_index, 1)
        x = x.mul(mask) 
        return x 
    
    # uniform drop channel
    def uniform_drop_channel(self, x, interval = 1):
        if not self.training:
            return self._triuvec(self.cov(x))     
        mask = paddle.zeros(size = (x.shape[1],)).cuda()
        if interval == 1:
            random_idx = paddle.randint(0,2, (1,))
            mask[random_idx::2] = 1
        elif interval == 2:
            random_idx = paddle.randint(0,3, (1,))
            mask[random_idx::3] = 1
        x = x.multiply(mask.reshape(-1,1,1))
        x = self._triuvec(self.cov(x))
        return x   

    # dynamic drop rate
    def balance_rate(self, x, y):
        y = paddle.flatten(y, 1) 
        d = x.shape[1]
        cor = paddle.sum(self.cov(x), axis=2)
        cor_, y_ = paddle.norm(cor, axis=1, p=2), paddle.norm(y, axis=1, p=2)
        rate = paddle.multiply(y, cor).sum(axis=1) / (y_ * cor_)   
        b_rate = 1 - (10/paddle.to_tensor([d], dtype='float32').log2()) * (rate.mean()) 
        b_rate = paddle.clip(b_rate, 0, 1)     
        model = nn.Dropout2D(b_rate.item()) 
        x = model(x)
        return x      


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)
         
        if self.training :
            x, y = self.eca_layer(x)
            x = self.balance_rate(x, y)

        x = self.cov(x)
        x = self._triuvec(x)
        
        x = paddle.flatten(x, 1)
        x = self.fc(x)
    
        return x 
       
def _resnet_ACD(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_ACD(block, layers, **kwargs)
    return model


def resnet18_ACD(pretrained=False, progress=True, **kwargs):
    return _resnet_ACD('resnet18_ACD', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34_ACD(pretrained=False, progress=True, **kwargs):
    return _resnet_ACD('resnet34_ACD', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_ACD(pretrained=False, progress=True, **kwargs):
    return _resnet_ACD('resnet50_ACD', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101_ACD(pretrained=False, progress=True, **kwargs):
    return _resnet_ACD('resnet101_ACD', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152_ACD(pretrained=False, progress=True, **kwargs):
    return _resnet_ACD('resnet152_ACD', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)



if __name__ == '__main__':
    model = resnet18_ACD()
    x = paddle.randn((2, 3, 224, 224))
    y = model(x)
    print(y.shape)