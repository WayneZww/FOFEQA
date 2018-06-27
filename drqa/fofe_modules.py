import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .modules import bn_conv, depthwise_conv_bn, SelfAttention


class fofe_conv1d(nn.Module):
    def __init__(self,
                 emb_dims,
                 alpha=0.9,
                 length=1,
                 dilation=1,
                 inverse=False):
        super(fofe_conv1d, self).__init__()
        self.alpha = alpha
        self.length = length
        self.channels = emb_dims
        self.fofe_filter = Parameter(torch.Tensor(emb_dims, 1, length))
        self.fofe_filter.requires_grad_(False)
        self._init_filter(emb_dims, alpha, length, inverse)
        self.padding = (length - 1) // 2
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(
                self.channels,
                self.channels,
                3,
                1,
                padding=length,
                dilation=dilation,
                groups=1,
                bias=False),
            nn.ReLU(inplace=True))

    def _init_filter(self, channels, alpha, length, inverse):
        if not inverse:
            self.fofe_filter[:, :, ].copy_(
                torch.pow(self.alpha, torch.linspace(length - 1, 0, length)))
        else:
            self.fofe_filter[:, :, ].copy_(
                torch.pow(self.alpha, torch.range(0, length - 1)))

    def forward(self, x):
        x = torch.transpose(x, -2, -1)
        if (self.length % 2 == 0):
            x = F.pad(x, (0, 1), mode='constant', value=0)
        x = F.conv1d(
            x,
            self.fofe_filter,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=self.channels)
        x = self.dilated_conv(x)
        return x


class fofe_filter(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = alpha
        self.fofe_filter = Parameter(torch.Tensor(inplanes, 1, length))
        self.fofe_filter.requires_grad_(False)
        self.inverse = inverse
        if inverse:
            self.padding = (0, self.length - 1)
        else:
            self.padding = (self.length - 1, 0)
        self._init_filter(alpha, length, inverse)

    def _init_filter(self, alpha, length, inverse):
        if not inverse:
            self.fofe_filter[:, :, ].copy_(
                torch.pow(alpha, torch.linspace(length - 1, 0, length)))
        else:
            self.fofe_filter[:, :, ].copy_(
                torch.pow(alpha, torch.range(0, length - 1)))

    def fofe_encode(self, x):
        out = F.pad(x, self.padding, mode='constant', value=0)
        out = F.conv1d(
            out,
            self.fofe_filter,
            bias=None,
            stride=1,
            padding=0,
            groups=self.channels)
        return out

    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0:
            return x
        out = self.fofe_encode(x)
        return out

    def extra_repr(self):
        return 'channels={channels}, alpha={alpha}, length={length}， ' \
            'inverse={inverse}, pad={padding}'.format(**self.__dict__)


class fofe_res_filter(fofe_filter):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_res_filter, self).__init__(inplanes, alpha, length, inverse)

    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0:
            return x
        residual = x
        out = self.fofe_encode(x)
        out += residual
        return out


class fofe_linear_res(fofe_filter):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_linear_res, self).__init__(inplanes, alpha, length,
                                                     inverse)
        self.W = nn.Conv1d(inplanes, inplanes, 1, 1, bias=False)
        nn.init.constant_(self.W.weight, 0)
        #nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0:
            return x
        residual = x
        out = self.fofe_encode(x)
        out = self.W(out)
        out += residual
        return out


class fofe_bi_res(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3):
        super(fofe_bi_res, self).__init__()
        self.forward_filter = fofe_filter(inplanes, alpha, length, False)
        self.inverse_filter = fofe_filter(inplanes, alpha, length, True)
        self.W = nn.Conv1d(inplanes * 2, inplanes, 1, 1, bias=False)

    #  nn.init.constant_(self.W.weight, 0)

    def forward(self, x):
        residual = x
        fofe_code = []
        fofe_code.append(self.forward_filter(x))
        fofe_code.append(self.inverse_filter(x))
        fofe_code = torch.cat(fofe_code, dim=1)
        out = self.W(fofe_code)
        out += residual
        return out


class res_block(nn.Module):
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def res_conv(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out

    def forward(self, x):
        x = self.res_conv(x)
        return x


class res_conv_block(res_block):
    def __init__(self, inplanes, planes, convs=3, dilation=1, downsample=None):
        super(res_conv_block, self).__init__()
        self.conv = []
        self.conv.append(
            bn_conv(
                inplanes,
                planes,
                3,
                1,
                dilation,
                dilation,
                groups=1,
                bias=False))

        for i in range(1, convs):
            self.conv.append(nn.ReLU(inplace=True))
            self.conv.append(
                bn_conv(
                    planes,
                    planes,
                    3,
                    1,
                    dilation,
                    dilation,
                    groups=1,
                    bias=False))

        self.conv = nn.Sequential(*self.conv)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.apply(self.weights_init)


class fofe_res_conv_block(res_conv_block):
    def __init__(self,
                 inplanes,
                 planes,
                 convs=3,
                 fofe_alpha=0.9,
                 fofe_length=3,
                 dilation=1,
                 downsample=None,
                 fofe_inverse=False):
        super(fofe_res_conv_block, self).__init__(inplanes, planes, convs,
                                                  dilation, downsample)
        self.fofe_filter = fofe_res_filter(inplanes, fofe_alpha, fofe_length,
                                           fofe_inverse)

    def forward(self, x):
        x = self.fofe_filter(x)
        x = self.res_conv(x)
        return x
    
    
class fofe_linear_res_block(res_conv_block):
    def __init__(self,
                 inplanes,
                 planes,
                 convs=3,
                 fofe_alpha=0.9,
                 fofe_length=3,
                 dilation=1,
                 downsample=None,
                 fofe_inverse=False):
        super(fofe_linear_res_block, self).__init__(inplanes, planes, convs,
                                                  dilation, downsample)
        self.fofe_filter = fofe_linear_res(inplanes, fofe_alpha, fofe_length,
                                           fofe_inverse)

    def forward(self, x):
        x = self.fofe_filter(x)
        x = self.res_conv(x)
        return x


class fofe_bi_res_block(res_conv_block):
    def __init__(self,
                 inplanes,
                 planes,
                 convs=3,
                 fofe_alpha=0.9,
                 fofe_length=3,
                 dilation=1,
                 downsample=None):
        super(fofe_bi_res_block, self).__init__(inplanes, planes, convs,
                                                dilation, downsample)
        self.fofe_filter = fofe_bi_res(inplanes, fofe_alpha, fofe_length)

    def forward(self, x):
        x = self.fofe_filter(x)
        x = self.res_conv(x)
        return x


class fofe_depthwise_res_block(res_block):
    def __init__(self,
                 inplanes,
                 planes,
                 convs=3,
                 fofe_alpha=0.9,
                 fofe_length=3,
                 dilation=1,
                 downsample=None):
        super(fofe_depthwise_res_block, self).__init__()
        self.conv = []
        self.conv.append(
            depthwise_conv_bn(
                inplanes, planes, 3, 1, dilation, dilation, bias=False))

        for i in range(1, convs):
            self.conv.append(nn.ReLU(inplace=True))
            self.conv.append(
                depthwise_conv_bn(
                    planes, planes, 3, 1, dilation, dilation, bias=False))

        self.conv = nn.Sequential(*self.conv)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.apply(self.weights_init)
        self.fofe_filter = fofe_bi_res(inplanes, fofe_alpha, fofe_length)

    def forward(self, x):
        x = self.fofe_filter(x)
        x = self.res_conv(x)
        return x


class fofe_res_att_block(fofe_res_conv_block):
    def __init__(self,
                 inplanes,
                 planes,
                 convs=3,
                 fofe_alpha=0.9,
                 fofe_length=3,
                 dilation=1,
                 downsample=None,
                 fofe_inverse=False):
        super(fofe_res_att_block,
              self).__init__(inplanes, planes, convs, fofe_alpha, fofe_length,
                             dilation, downsample, fofe_inverse)
        self.att = SelfAttention(planes)

    def forward(self, x):
        x = self.fofe_filter(x)
        x = self.res_conv(x)
        x = self.att(x)
        return x
    
    
class fofe_linear_res_att_block(fofe_linear_res_block):
    def __init__(self,
                 inplanes,
                 planes,
                 convs=3,
                 fofe_alpha=0.9,
                 fofe_length=3,
                 dilation=1,
                 downsample=None,
                 fofe_inverse=False):
        super(fofe_linear_res_att_block,
              self).__init__(inplanes, planes, convs, fofe_alpha, fofe_length,
                             dilation, downsample, fofe_inverse)
        self.att = SelfAttention(planes)

    def forward(self, x):
        x = self.fofe_filter(x)
        x = self.res_conv(x)
        x = self.att(x)
        return x


class fofe_linear(nn.Module):
    def __init__(self, channels, alpha):
        super(fofe_linear, self).__init__()
        self.alpha = alpha
        self.linear = nn.Sequential(
            nn.Linear(channels, channels, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        length = x.size(-2)
        #Should use new_empty here
        matrix = x.new_empty(x.size(0), 1, length)
        #if x.data.is_cuda :
        #    matrix = matrix.cuda()
        matrix[:, ].copy_(
            torch.pow(self.alpha, torch.linspace(length - 1, 0, length)))
        fofe_code = torch.bmm(matrix, x)
        output = self.linear(fofe_code)
        return output


class fofe(nn.Module):
    def __init__(self, channels, alpha):
        super(fofe, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        length = x.size(-2)
        matrix = x.new_empty(x.size(0), 1, length)
        matrix[:, ].copy_(
            torch.pow(self.alpha, torch.linspace(length - 1, 0, length)))
        fofe_code = torch.bmm(matrix, x)
        return fofe_code
