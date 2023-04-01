import paddle
from paddle.fluid.layers.loss import cross_entropy
from paddle.fluid.layers.nn import shape 
import paddle.nn as nn 
from paddle.autograd import PyLayer 
import numpy as np


class MPNCOV(nn.Layer): 
    
     def __init__(self, iterNum=3, is_sqrt=True, is_vec=True, input_dim=2048, dimension_reduction=None):
    
         super(MPNCOV, self).__init__()
         self.iterNum=iterNum
         self.is_sqrt = is_sqrt
         self.is_vec = is_vec
         self.dr = dimension_reduction
         if self.dr is not None:
             self.conv_dr_block = nn.Sequential(
               nn.Conv2D(input_dim, self.dr, kernel_size=1, stride=1, bias_attr=None),
               nn.BatchNorm2D(self.dr),
               nn.ReLU()
             )
         output_dim = self.dr if self.dr else input_dim
         if self.is_vec:
             self.output_dim = int(output_dim*(output_dim+1)/2)
         else:
             self.output_dim = int(output_dim*output_dim)
         self._init_weight()   

     def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0]*m.weight.shape[1]*m.weight.shape[2]
                v = np.random.normal(loc=0.,scale=np.sqrt(2./n),size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
    
     def _cov_pool(self, x):
         return Covpool.apply(x)
     def _sqrtm(self, x):
         return Sqrtm.apply(x, self.iterNum)
     def _triuvec(self, x):
         return Triuvec.apply(x)    

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         x = self._cov_pool(x)
         if self.is_sqrt:
             x = self._sqrtm(x)
         if self.is_vec:
             x = self._triuvec(x)
         return x

class Covpool(PyLayer):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.shape[0]
         dim = x.shape[1]
         h = x.shape[2]
         w = x.shape[3]
         M = h*w
         x = x.reshape([batchSize,dim,M])
         I_hat = (-1./M/M)*paddle.ones(shape=[M,M]) + (1./M)*paddle.eye(M,M)
         I_hat = paddle.fluid.layers.expand(I_hat.reshape([1,M,M]), expand_times=[batchSize,1,1]).astype(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose([0, 2, 1]))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensor()
         x = input
         batchSize = x.shape[0]
         dim = x.shape[1]
         h = x.shape[2]
         w = x.shape[3]
         M = h*w
         x = x.reshape([batchSize,dim,M])
         grad_input = grad_output + grad_output.transpose([0, 2, 1])
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape([batchSize,dim,h,w])
         return grad_input


class Sqrtm(PyLayer):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input 
        batchSize = x.shape[0]
        dim = x.shape[1]
        dtype = x.dtype
        I3 = paddle.fluid.layers.expand(3.0*paddle.eye(dim,dim).reshape([1, dim, dim]), expand_times=[batchSize,1,1]).astype(dtype)
        normA = (1.0/3.0)*x.multiply(I3).sum(axis=1).sum(axis=1)
        A = x.divide(normA.reshape([batchSize,1,1]).expand_as(x))
        Y = paddle.zeros(shape=[batchSize, iterN, dim, dim]).astype(dtype)
        Z = paddle.fluid.layers.expand(paddle.eye(dim,dim).reshape([1,1,dim,dim]), expand_times=[batchSize,iterN,1,1]).astype(x.dtype)
        if iterN < 2:
            ZY = 0.5*(I3 - A)
            YZY = A.bmm(ZY)   
        else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY             
            for i in range(1, iterN-1):
                ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
                Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
                Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            YZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
        y = YZY*paddle.sqrt(normA).reshape([batchSize, 1, 1]).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y   
    @staticmethod
    def backward(ctx, grad_output):             
        input, A, ZY, normA, Y, Z = ctx.saved_tensor()
        iterN = ctx.iterN
        x = input
        batchSize = x.shape[0]
        dim = x.shape[1]
        dtype = x.dtype 
        der_postCom = grad_output*paddle.sqrt(normA).reshape([batchSize, 1, 1]).expand_as(x)       
        der_postComAux = (grad_output*ZY).sum(axis=1).sum(axis=1).divide(2*paddle.sqrt(normA))
        I3 = paddle.fluid.layers.expand(3.0*paddle.eye(dim,dim).reshape([1, dim, dim]), expand_times=[batchSize,1,1]).astype(dtype)
        if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
                YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
                ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
                dldY_ = 0.5*(dldY.bmm(YZ) -
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                             ZY.bmm(dldY))
                dldZ_ = 0.5*(YZ.bmm(dldZ) -
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose([0, 2, 1])
        grad_input_ = der_NSiter.divide(normA.reshape([batchSize,1,1]).expand_as(x))
        grad_aux = der_NSiter.multiply(x).sum(axis=1).sum(axis=1)
        for i in range(batchSize):
            grad_input_[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *paddle.diag(paddle.ones(shape=[dim])).astype(dtype)
        grad_input = grad_input_ + 0
        return grad_input

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

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def TriuvecLayer(var):
    return Triuvec.apply(var)

if __name__ == '__main__':
    data = paddle.randn([1,3,5,5])
    data.stop_gradient = False
    label = paddle.to_tensor([1])

    x = CovpoolLayer(data)
    x = SqrtmLayer(x, 5)
    y = TriuvecLayer(x)
    print(y.shape)
    y.mean().backward()
    print(data.grad)




