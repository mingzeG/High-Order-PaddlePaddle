'''
@file: bdc_modele.py
@author: Fei Long
@author: Jiaming Lv
Please cite the paper below if you use the code:

Jiangtao Xie, Fei Long, Jiaming Lv, Qilong Wang and Peihua Li. Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.

Copyright (C) 2022 Fei Long and Jiaming Lv

All rights reserved.
'''

import paddle
import paddle.nn as nn

class BDC(nn.Layer):
    def __init__(self, is_vec=True, input_dim=640, dimension_reduction=None, activate='relu'):
        super(BDC, self).__init__()
        self.is_vec = is_vec
        self.dr = dimension_reduction
        self.activate = activate
        self.input_dim = input_dim
        if self.dr is not None and self.dr != self.input_dim:
            if activate == 'relu':
                self.act = nn.ReLU()
            elif activate == 'leaky_relu':
                self.act = nn.LeakyReLU(0.1)
            else:
                self.act = nn.ReLU()

            self.conv_dr_block = nn.Sequential(
            nn.Conv2D(self.input_dim, self.dr, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2D(self.dr),
            self.act
            )
        output_dim = self.dr if self.dr else self.input_dim
        if self.is_vec:
            self.output_dim = int(output_dim*(output_dim+1)/2)
        else:
            self.output_dim = int(output_dim*output_dim)

        self.temperature = paddle.create_parameter((1,1), dtype='float32', default_initializer=nn.initializer.Constant(paddle.log((1. / (2 * input_dim*input_dim)) * paddle.ones((1,1)))))
        # paddle.log((1. / (2 * input_dim*input_dim)) * paddle.ones((1,1))
        self._init_weight()

    def _init_weight(self):
        for m in self.named_parameters():
            if isinstance(m, nn.Conv2D):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2D):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.dr is not None and self.dr != self.input_dim:
            x = self.conv_dr_block(x)
        x = BDCovpool(x, self.temperature)
        if self.is_vec:
            x = Triuvec(x)
        else:
            x = x.reshape((x.shape[0], -1))
        return x

def BDCovpool(x, t):
    batchSize, dim, h, w = x.shape
    M = h * w
    x = x.reshape((batchSize, dim, M))

    I = paddle.fluid.layers.expand(paddle.eye(dim, dim).reshape((1, dim, dim)), expand_times=[batchSize, 1, 1])
    I_M = paddle.ones((batchSize, dim, dim))
    x_pow2 = x.bmm(x.transpose((0, 2, 1)))
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    
    dcov = paddle.clip(dcov, min=0.0)
    dcov = paddle.exp(t)* dcov
    dcov = paddle.sqrt(dcov + 1e-5)
    t = dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)

    return t


def Triuvec(x):
    batchSize, dim, dim = x.shape
    x = x.reshape((batchSize, dim * dim)).numpy()
    I = paddle.triu(paddle.ones(shape=[dim,dim])).reshape([dim*dim])
    index = I.nonzero()
    temp = paddle.to_tensor(x[:,index])
    y = paddle.zeros([batchSize,int(dim*(dim+1)/2)])
    y = paddle.to_tensor(temp)
    return y

# if __name__ =='__main__':
#     x = paddle.ones(shape=(2,3,224,224))
#     model = BDC()
#     y = model(x)