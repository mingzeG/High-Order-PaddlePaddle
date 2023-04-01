from paddle import nn
import paddle
import pdb
from basic_ops import ConsensusModule
from transforms import *
from paddle.nn.initializer import Normal, Constant
from TCP.TCP_module import TCP


# from ops.paddlevision import  res_224 as rsn #gzl 01.18

class TSN(nn.Layer):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet50', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False,
                 # convgru=False, deformconvgru=False,
                 # ctconvgru=False, ctdeformconvgru=False,
                 TCP_level = 'video',
                 TCP=False, TCP_dim=0, pretrained_dim=None,
                 TCP_ch=None, TCP_sp=None, TCP_1D=None,
                 temporal_nonlocal=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool

        self.TCP = TCP
        self.TCP_dim = TCP_dim
        self.TCP_level = TCP_level
        self.pretrained_dim = pretrained_dim
        self.TCP_ch = TCP_ch
        self.TCP_sp = TCP_sp
        self.TCP_1D = TCP_1D

        self.temporal_nonlocal = temporal_nonlocal

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_Layer:   {}
        dropout_ratio:      {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        if self.TCP :
            feature_dim = self.repr_dim
        else :
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).weight.shape[0]
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class, weight_attr=Normal(0, 0.001),
                   bias_attr=Constant(value=0))

        # std = 0.001
        # if self.new_fc is None:
        #     Normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
        #     Constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
  
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if self.TCP:
            TCP_Layer = TCP(
                dim_in=2048, dim_out=self.TCP_dim,
                num_segments=self.num_segments,
                ch_flag=self.TCP_ch, sp_flag=self.TCP_sp,)
                # conv_1D_flag=self.TCP_1D)
            self.repr_dim = int(self.TCP_dim * (self.TCP_dim + 1) / 2.)
        else:
            TCP_Layer = None
            self.repr_dim = 2048


        if 'resnet152' in base_model : #Preact ResNet-152 pretrained on ImageNet-11K + Imagenet-1K
            from paddlevision import preact_resnet as rsn
            self.base_model = getattr(rsn, base_model)(True,
                                                       TCP=TCP_Layer)  # gzl 12.01

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length


        elif 'resnet' in base_model:
            from paddlevision import resnet as rsn
            self.base_model = getattr(rsn, base_model)(False,
                                                       TCP=TCP_Layer)  # gzl 12.01

            if self.is_shift:
                print('Adding temporal shift...')
                from temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif 'tea' in base_model:

            if self.num_segments == 8:
                from paddlevision.tea50_8f import tea50_8f
                self.base_model = tea50_8f(
                    TCP_module=TCP_Layer,
                )
            if self.num_segments == 16:
                from paddlevision.tea50_16f import tea50_16f
                self.base_model = tea50_16f(
                    TCP_module=TCP_Layer,
                )

            self.input_size = 224
            self.base_model.last_layer_name = 'fc'

            if self.modality == 'RGB':
                self.input_mean = [0.485, 0.456, 0.406]
                self.input_std = [0.229, 0.224, 0.225]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)

        if mode:
            if 'resnet152' in self.base_model_name :
                self.base_model.bn_data.weight.requires_grad=  False
                print('----freeze input bn weight for preact model-----------')

        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.Layers():
                if isinstance(m, nn.BatchNorm2D):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self, args):
        first_conv_weight = []
        first_conv_bias = []
        Normalweight = []
        Normalbias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        ln = []
        TCP_weight = []
        TCP_bias = []
        TCP_bn = []

        conv_cnt = 0
        bn_cnt = 0
        ln_cnt = 0
        TCP_FLAG = False
        for m in self.Layers():
            if m == self.new_fc :
                TCP_FLAG = False
            if hasattr(self.base_model, 'TCP') :
                if self.pretrained_dim != self.TCP_dim :
                    if m == self.base_model.TCP.layer_reduce2:
                        TCP_FLAG = True
                else :
                    if m == self.base_model.TCP.TCP_att:
                        TCP_FLAG = True

            # print('FOR m:{}, TCP_FLAG:{}'.format(m, TCP_FLAG))

            if isinstance(m, paddle.nn.Conv2D) or isinstance(m, paddle.nn.Conv1D) \
                    or isinstance(m, paddle.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                elif TCP_FLAG :
                    TCP_weight.append(ps[0])
                    if len(ps) == 2:
                        TCP_bias.append(ps[1])
                else:
                    Normalweight.append(ps[0])
                    if len(ps) == 2:
                        Normalbias.append(ps[1])

            elif isinstance(m, paddle.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    Normalweight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        Normalbias.append(ps[1])

            elif isinstance(m, paddle.nn.LayerNorm):
                ln_cnt += 1
                # if ln_cnt == 1:
                if TCP_FLAG :
                    TCP_bn.extend(list(m.parameters()))
                else :
                    ln.extend(list(m.parameters()))

            elif isinstance(m, paddle.nn.BatchNorm2D) \
                    or isinstance(m, paddle.nn.BatchNorm3d) \
                    or isinstance(m, paddle.nn.SyncBatchNorm):
                bn_cnt += 1
                if TCP_FLAG:
                    TCP_bn.extend(list(m.parameters()))
                elif not self._enable_pbn:
                        bn.extend(list(m.parameters()))
            elif len(m._Layers) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic Layer type: {}. Need to give it a learning policy".format(type(m)))

        lr_x_factor = args.lr_x_factor

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': Normalweight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "Normalweight"},
            {'params': Normalbias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "Normalbias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': ln, 'lr_mult': 1, 'decay_mult': 0,
             'name': "LN scale/shift"},

            # TCP Layer
            {'params': TCP_weight, 'lr_mult': lr_x_factor, 'decay_mult': 1,
             'name': "TCP_weight"},
            {'params': TCP_bias, 'lr_mult': lr_x_factor, 'decay_mult': 0,
             'name': "TCP_bias"},
            {'params': TCP_bn, 'lr_mult': lr_x_factor, 'decay_mult': 1,
             'name': "TCP_bn scale/shift"},

            # for fc
            {'params': lr5_weight, 'lr_mult': lr_x_factor, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 2*lr_x_factor, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]


    def forward(self, input, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            # print((-1, sample_len) + input.shape()[-2:])
            base_out = self.base_model(input.reshape((-1, sample_len) + tuple(input.shape[-2:])))
        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + tuple(base_out.shape[1:]))
            elif self.TCP and self.TCP_level == 'video':
                base_out = base_out
            else :
                base_out = base_out.reshape((-1, self.num_segments) + tuple(base_out.shape[1:]))
            if (not self.TCP) or (self.TCP and self.TCP_level=='frame'):
                output = self.consensus(base_out)
                return output.squeeze(1)
            else :
                return base_out

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # paddle models are usually defined in a hierarchical way.
        # nn.Layers.children() return all sub Layers in a DFS manner
        Layers = list(self.base_model.Layers())
        first_conv_idx = list(filter(lambda x: isinstance(Layers[x], nn.Conv2D), list(range(len(Layers)))))[0]
        conv_layer = Layers[first_conv_idx]
        container = Layers[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2D(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        # if self.base_model_name == 'BNInception':
        #     import paddle.utils.model_zoo as model_zoo
        #     sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
        #     base_model.load_state_dict(sd)
        #     print('=> Loading pretrained Flow weight done...')
        # else:
        #     print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # paddle models are usually defined in a hierarchical way.
        # nn.Layers.children() return all sub Layers in a DFS manner
        Layers = list(self.base_model.Layers())
        first_conv_idx = filter(lambda x: isinstance(Layers[x], nn.Conv2D), list(range(len(Layers))))[0]
        conv_layer = Layers[first_conv_idx]
        container = Layers[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = paddle.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2D(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return paddle.vision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return paddle.vision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return paddle.vision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return paddle.vision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

if __name__ == '__main__':
    model = TSN(num_class=10, num_segments=8, modality='RGB')
    x = paddle.randn((8, 8, 3, 224, 224))
    y = model(x)