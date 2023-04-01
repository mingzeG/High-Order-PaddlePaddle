import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer
import numpy as np

class Covariance(nn.Layer):
    def __init__(self, 
                cov_type='norm',
                remove_mean=True,
                dimension_reduction=None,
                input_dim=2048,
        ):
        super(Covariance, self).__init__()
        self.cov_type = cov_type
        self.remove_mean = remove_mean
        self.dr = dimension_reduction
        if self.dr is not None:
            if self.cov_type == 'norm':
                self.conv_dr_block = nn.Sequential(
                    nn.Conv2D(input_dim, self.dr[0], kernel_size=1, stride=1, bias_attr=False),
                    nn.BatchNorm2d(self.dr[0]),
                    nn.ReLU(inplace=True)
                )
            elif self.cov_type == 'cross':
                self.conv_dr_block = nn.Sequential(
                    nn.Sequential(
                        nn.Conv2D(input_dim, self.dr[0], kernel_size=1, stride=1, bias_attr=False),
                        nn.BatchNorm2d(self.dr[0]),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Sequential(
                        nn.Conv2D(input_dim, self.dr[1], kernel_size=1, stride=1, bias_attr=False),
                        nn.BatchNorm2d(self.dr[1]),
                        nn.ReLU(inplace=True)
                    )
                )
        self._init_weight()

    def _init_weight(self):
        for m in self.parameters():
            if isinstance(m, nn.Conv2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

    def _remove_mean(self, x):
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        return x

    def _cov(self, x):
        # channel
        batchsize, d, h, w = x.shape
        N = h*w
        x = x.reshape((batchsize, d, N))
        y = (1. / N ) * (x.bmm(x.transpose((0, 2, 1))))
        return y
    
    def _cross_cov(self, x1, x2):
        # channel
        batchsize1, d1, h1, w1 = x1.size()
        batchsize2, d2, h2, w2 = x2.size()
        N1 = h1*w1
        N2 = h2*w2
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.view(batchsize1, d1, N1)
        x2 = x2.view(batchsize2, d2, N2)

        y = (1. / N1) * (x1.bmm(x2.transpose(0, 2, 1)))
        return y
    
    def forward(self, x, y=None):
        #import pdb;pdb.set_trace()
        if self.dr is not None:
            if self.cov_type == 'norm':
                x = self.conv_dr_block(x)
            elif self.cov_type == 'cross':
                if y is not None:
                    x = self.conv_dr_block[0](x)
                    y = self.conv_dr_block[1](y)
                else:
                    ori = x
                    x = self.conv_dr_block[0](ori)
                    y = self.conv_dr_block[1](ori)
        if self.remove_mean:
            x = self._remove_mean(x)
            if y is not None:
                y = self._remove_mean(y)               
        if y is not None:
            x = self._cross_cov(x, y)
        else:
            x = self._cov(x)
        return x

class Triuvec(PyLayer):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.shape[0]
        dim = x.shape[1]
        dtype = x.dtype
        x = x.reshape([batchSize, dim*dim]).numpy()
        I = paddle.triu(paddle.ones(shape=[dim,dim])).reshape([dim*dim])
        index = I.nonzero()
        temp = paddle.to_tensor(x[:,index])
        y = paddle.zeros([batchSize,int(dim*(dim+1)/2)]).astype(dtype)
        y = paddle.to_tensor(temp)
        ctx.save_for_backward(input,index)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        input,index = ctx.saved_tensor()
        x = input
        batchSize = x.shape[0]
        dim = x.shape[1]
        dtype = x.dtype
        grad_input = paddle.zeros([batchSize,dim*dim]).astype(dtype).numpy()
        grad_input[:,index] = grad_output.numpy()
        grad_input = paddle.to_tensor(grad_input.reshape([batchSize,dim,dim]))
        return grad_input