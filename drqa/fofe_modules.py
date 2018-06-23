import math
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class fofe_conv1d(nn.Module):
    def __init__(self, emb_dims, alpha=0.9, length=1, dilation=1, inverse=False):
        super(fofe_conv1d, self).__init__()
        self.alpha = alpha
        self.length = length
        self.channels = emb_dims
        self.fofe_filter = Parameter(torch.Tensor(emb_dims,1,length))
        self.fofe_filter.requires_grad_(False)
        self._init_filter(emb_dims, alpha, length, inverse)
        self.padding = (length - 1)//2
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(self.channels,self.channels,3,1,padding=length,
                        dilation=dilation, groups=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)           
        )

    def _init_filter(self, channels, alpha, length, inverse):
        if not inverse :
            self.fofe_filter[:,:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        else :
            self.fofe_filter[:,:,].copy_(torch.pow(self.alpha,torch.range(0,length-1)))

    def forward(self, x): 
        x = torch.transpose(x,-2,-1)
        if (self.length % 2 == 0) :
            x = F.pad(x, (0,1), mode='constant', value=0)
        x = F.conv1d(x, self.fofe_filter, bias=None, stride=1, 
                        padding=self.padding, groups=self.channels)
        x = self.dilated_conv(x)
        return x

    
class fofe_filter(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = alpha
        self.fofe_filter = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_filter.requires_grad_(False)
        self.inverse = inverse
        if inverse :
            self.padding = (0, self.length-1)
        else :
            self.padding = (self.length-1, 0)
        self._init_filter(alpha, length, inverse)

    def _init_filter(self, alpha, length, inverse):
        if not inverse :
            self.fofe_filter[:,:,].copy_(torch.pow(alpha,torch.linspace(length-1,0,length)))
        else :
            self.fofe_filter[:,:,].copy_(torch.pow(alpha,torch.range(0,length-1)))
    
    def fofe_encode(self, x):
        out = F.pad(x, self.padding, mode='constant', value=0)
        out = F.conv1d(out, self.fofe_filter, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        return out
        
    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0 :
            return x
        x = self.fofe_encode(x)
        return 
    
    def extra_repr(self):
        return 'channels={channels}, alpha={alpha}, length={length}， ' \
            'inverse={inverse}, pad={padding}'.format(**self.__dict__)


class fofe_res_filter(fofe_filter):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_res_filter, self).__init__(inplanes, alpha, length, inverse)
    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0 :
            return x
        residual = x
        out = self.fofe_encode(x)
        out += residual
        return out
      
class ln_conv(nn.Module):
    def __init__(self,inplanes, planes, kernel, stride, 
                        padding=0, dilation=1, groups=1, bias=False):
        super(ln_conv, self).__init__()
        self.layer_norm = nn.LayerNorm(inplanes)
        self.conv = nn.Conv1d(inplanes, planes, kernel, stride, padding,
                                dilation, groups, bias=False)
    def forward(self, x):
        out = self.layer_norm(x.transpose(-1,-2)).transpose(-1,-2)
        out = self.conv(out)
        return out
    
class bn_conv(nn.Module):
    def __init__(self,inplanes, planes, kernel, stride, 
                        padding=0, dilation=1, groups=1, bias=False):
        super(bn_conv, self).__init__()
        self.conv = nn.Conv1d(inplanes, planes, kernel, stride, padding,
                                dilation, groups, bias=False)
        self.bn = nn.BatchNorm1d(planes)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out
    
class res_bn_conv(nn.Module):
    def __init__(self,inplanes, planes, kernel, stride, 
                        padding=0, dilation=1, groups=1, bias=False):
        super(res_bn_conv, self).__init__()
        self.conv = nn.Conv1d(inplanes, planes, kernel, stride, padding,
                                dilation, groups, bias=False)
        self.bn = nn.BatchNorm1d(planes)
        self.downsample = None
        if inplanes != planes :
            self.downsample = nn.Conv1d(inplanes, planes, 1, 1, 0,
                                1, groups=1, bias=False)
    def forward(self, x):
        residual = x
        if self.downsample != None :
            residual = self.downsample(x)
        out = self.conv(x)
        out = self.bn(out)
        out += residual
        return out

class fofe_block(nn.Module):
    def __init__(self, inplanes, planes, fofe_alpha=0.8, fofe_length=3, dilation=3, fofe_inverse=False):
        super(fofe_block, self).__init__()
        self,fofe_filter = fofe_filter(inplanes, fofe_alpha, fofe_length, fofe_inverse)
        self.conv = nn.Sequential(nn.Conv1d(inplanes, planes,3,1,padding=length,
                        dilation=dilation, groups=1, bias=False),
                    nn.BatchNorm1d(planes),
                    nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.fofe_filter(x)
        x = self.conv(x)
    
        return x
    

class fofe_res_block(nn.Module):
    def __init__(self, inplanes, planes, convs=3, fofe_alpha=0.9, fofe_length=3, 
                        dilation=1, downsample=None, fofe_inverse=False):
        super(fofe_res_block, self).__init__()
        self.fofe_filter = fofe_res_filter(inplanes, fofe_alpha, fofe_length, fofe_inverse)
        
        self.conv = []
        self.conv.append(nn.Sequential(
                            nn.Conv1d(inplanes, planes,3,1,dilation, dilation, groups=1, bias=False),
                            nn.BatchNorm1d(planes)))

        for i in range(1, convs):
            self.conv.append(nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                nn.Conv1d(planes, planes, 3, 1, dilation, dilation, groups=1, bias=False),
                                nn.BatchNorm1d(planes)))

        self.conv = nn.Sequential(*self.conv)
        self.relu = nn.LeakyReLU(0.1, inplace=True) 
        self.downsample = downsample
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, x): 
        x = self.fofe_filter(x)
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out

class fofe_res_conv_block(nn.Module):
    def __init__(self, inplanes, planes, convs=3, fofe_alpha=0.9, fofe_length=3, 
                        dilation=1, downsample=None, fofe_inverse=False):
        super(fofe_res_conv_block, self).__init__()
        self.fofe_filter = fofe_res_filter(inplanes, fofe_alpha, fofe_length, fofe_inverse)
        
        self.conv = []
        self.conv.append(bn_conv(inplanes, planes, 3, 1, dilation, 
                                 dilation, groups=1, bias=False))

        for i in range(1, convs):
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
            self.conv.append(bn_conv(planes, planes, 3, 1, dilation, 
                                     dilation, groups=1, bias=False))

        self.conv = nn.Sequential(*self.conv)
        self.relu = nn.LeakyReLU(0.1, inplace=True) 
        self.downsample = downsample
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, x): 
        x = self.fofe_filter(x)
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out


class fofe_linear(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe_linear, self).__init__()
        self.alpha = alpha
        self.linear = nn.Sequential(
            nn.Linear(channels,channels, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        length = x.size(-2)
        #Should use new_empty here
        matrix = x.new_empty(x.size(0),1,length)
        #if x.data.is_cuda :
        #    matrix = matrix.cuda()
        matrix[:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        fofe_code = torch.bmm(matrix,x)
        output = self.linear(fofe_code)
        return output


class fofe(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        length = x.size(-2)
        matrix = x.new_empty(x.size(0),1,length)
        matrix[:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        fofe_code = torch.bmm(matrix,x)
        return fofe_code


class ASPP(nn.Module):
    def __init__(self, planes, rates):
        super(ASPP, self).__init__()
        layers = []
        self.planes = planes
        self.rates = rates
        for v in rates:
            layers.append(nn.Sequential(
                bn_conv(planes, planes, 3, 1, v, v, groups=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True)))
        self.dilated_conv = nn.ModuleList(layers)
        self.aggregate = nn.Sequential(
                bn_conv(planes*len(rates), planes, 1, 1, 0, 1, groups=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True))
        
    def forward(self, x):
        out = []
        for layer in self.dilated_conv :
            out.append(layer(x))
        
        out = torch.cat(out, dim=1)
        out = self.aggregate(out)
        return out
    
    def extra_repr(self):
        return 'planes={planes}, rates={rates}'.format(**self.__dict__)



class Simility(nn.Module):
    def __init__(self, planes):
        super(Simility, self).__init__()
        self.W = nn.Conv2d(planes*3, 1, 1, 1, bias=False)
        self.W.weight.data = nn.init.kaiming_normal_(self.W.weight.data)

    def forward(self, doc, query):
        d_length = doc.size(-1)
        q_length = query.size(-1)
        d_matrix = []
        q_matrix = []
        a_matrix = []

        for i in range(q_length):
            d_matrix.append(doc.unsqueeze(-2))
        for j in range(d_length):
            q_matrix.append(query.unsqueeze(-1))
        d_matrix = torch.cat(d_matrix, dim=-2)
        q_matrix = torch.cat(q_matrix, dim=-1)
        s_matrix = d_matrix.mul(q_matrix)

        a_matrix.append(d_matrix)
        a_matrix.append(q_matrix)
        a_matrix.append(s_matrix)
        a_matrix = torch.cat(a_matrix, dim=1)
        simility = self.W(a_matrix).squeeze(1)

        return simility


class Attention(nn.Module):
    def __init__(self, planes):
        super(Attention, self).__init__()
        self.simility = Simility(planes)
    
    def forward(self, doc, query):
        simility = self.simility(doc, query)
        s1_t = F.softmax(simility,dim=-2).transpose(-1,-2)
        c2q_att = torch.bmm(s1_t, query.transpose(-1,-2)).transpose(-1,-2) #batchsize x d x n
        s2 = F.softmax(simility, dim=-1)
        q2c_att = torch.bmm(s1_t, torch.bmm(s2, doc.transpose(-1,-2))).transpose(-1,-2) #batchsize x d x n

        output = []
        output.append(doc)
        output.append(c2q_att)
        output.append(doc.mul(c2q_att))
        output.append(doc.mul(q2c_att))
        output = torch.cat(output, dim=1)

        return output


class BiAttention(nn.Module):
    def __init__(self, planes):
        super(BiAttention, self).__init__()
        self.simility = Simility(planes)
    
    def forward(self, doc, query):
        simility = self.simility(doc, query)
        s1_t = F.softmax(simility,dim=-2).transpose(-1,-2)
        s2 = F.softmax(simility, dim=-1)

        d_d2q_att = torch.bmm(s1_t, query.transpose(-1,-2)).transpose(-1,-2) #batchsize x d x n
        d_q2d_att = torch.bmm(s1_t, torch.bmm(s2, doc.transpose(-1,-2))).transpose(-1,-2) #batchsize x d x n

        q_q2d_att = torch.bmm(s2, doc.transpose(-1,-2)).transpose(-1,-2) #batchsize x d x m
        q_d2q_att = torch.bmm(s2, torch.bmm(s1_t, query.transpose(-1,-2))).transpose(-1,-2) #batchsize x d x m

        d_output = []
        d_output.append(doc)
        d_output.append(d_d2q_att)
        d_output.append(doc.mul(d_d2q_att))
        d_output.append(doc.mul(d_q2d_att))
        d_output = torch.cat(d_output, dim=1)

        q_output = []
        q_output.append(query)
        q_output.append(q_q2d_att)
        q_output.append(query.mul(q_q2d_att))
        q_output.append(query.mul(q_d2q_att))
        q_output = torch.cat(q_output, dim=1)

        return d_output, q_output


class SelfAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(SelfAttention, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.Wv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.Wq = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)
        self.Wk = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.in_channels)
        )
        self.Wv.weight.data = nn.init.kaiming_normal_(self.Wv.weight.data)
        self.Wq.weight.data = nn.init.kaiming_normal_(self.Wq.weight.data)
        self.Wk.weight.data = nn.init.kaiming_normal_(self.Wk.weight.data)
        nn.init.constant(self.W[1].weight, 0)
        nn.init.constant(self.W[1].bias, 0)

    def forward(self, x):
        v_x = self.Wv(x).transpose(-1, -2)
        q_x = self.Wq(x).transpose(-1, -2)
        k_x = self.Wk(x)
        s = torch.bmm(q_x, k_x)
        similirity = F.softmax(s, dim=-2)

        y = torch.matmul(similirity, v_x).transpose(-1,-2).contiguous()
        W_y = self.W(y)
        output = W_y + x

        return output




