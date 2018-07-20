import math
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .utils import tri_num, count_num_substring

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
        self.fofe_kernel = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_kernel.requires_grad_(False)
        self._init_kernel(alpha, length, inverse)
        #self.padding = (length - 1)//2
        self.inverse = inverse

    def _init_kernel(self, alpha, length, inverse):
        if not inverse :
            self.fofe_kernel[:,:,].copy_(torch.pow(alpha, torch.linspace(length-1, 0, length)))
        else :
            self.fofe_kernel[:,:,].copy_(torch.pow(alpha, torch.range(0, length-1)))
    
    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0 :
            return x
        if self.inverse:
            x = F.pad(x,(0, self.length))
        else :
            x = F.pad(x,(self.length, 0))
        x = F.conv1d(x, self.fofe_kernel, bias=None, stride=1, 
                        padding=0, groups=self.channels)

        return x


class fofe_dual_filter(nn.Module):
    def __init__(self, inplanes, alpha_l=0.4, alpha_h=0.8, length=3, inverse=False):
        super(fofe_dual_filter, self).__init__()
        self.length = length
        self.inplanes = inplanes
        self.alpha_l = alpha_l
        self.alpha_h = alpha_h
        self.fofe_kernel_s = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_kernel_s.requires_grad_(False)
        self.fofe_kernel_l = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_kernel_l.requires_grad_(False)
        self._init_kernel(alpha_l, alpha_h, length, inverse)
        #self.padding = (length - 1)//2
        self.inverse = inverse

    def _init_kernel(self, alpha_l, alpha_h, length, inverse):
        if not inverse :
            self.fofe_kernel_s[:,:,].copy_(torch.pow(alpha_l, torch.linspace(length-1, 0, length)))
            self.fofe_kernel_l[:,:,].copy_(torch.pow(alpha_h, torch.linspace(length-1, 0, length)))
        else :
            self.fofe_kernel_s[:,:,].copy_(torch.pow(alpha_l, torch.range(0, length-1)))
            self.fofe_kernel_l[:,:,].copy_(torch.pow(alpha_h, torch.range(0, length-1)))
    
    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0 :
            return x
        if self.inverse:
            x = F.pad(x,(0, self.length))
        else :
            x = F.pad(x,(self.length, 0))
        short_fofe = F.conv1d(x, self.fofe_kernel_s, bias=None, stride=1, 
                        padding=0, groups=self.inplanes)
        long_fofe = F.conv1d(x, self.fofe_kernel_l, bias=None, stride=1, 
                        padding=0, groups=self.inplanes)
        fofe_code = torch.cat([short_fofe, long_fofe], dim=1)
        return fofe_code

    def extra_repr(self):
        return 'inplanes={inplanes}, alpha_l={alpha_l}, alpha_h={alpha_h}, ' \
                'length={length}, inverse={inverse}'.format(**self.__dict__)


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
        fofe_code = torch.bmm(matrix,x).squeeze(-2)
        return fofe_code

class fofe_dual(nn.Module):
    def __init__(self, channels, alpha_l, alpha_h): 
        super(fofe_dual, self).__init__()
        self.alpha_l = alpha_l
        self.alpha_h = alpha_h
        
    def forward(self, x):
        length = x.size(-2)
        matrix_s = x.new_empty(x.size(0),1,length)
        matrix_s[:,].copy_(torch.pow(self.alpha_l,torch.linspace(length-1,0,length)))
        matrix_l = x.new_empty(x.size(0),1,length)
        matrix_l[:,].copy_(torch.pow(self.alpha_h,torch.linspace(length-1,0,length)))
        short_fofe = torch.bmm(matrix_s,x).squeeze(-2)
        long_fofe = torch.bmm(matrix_l,x).squeeze(-2)
        fofe_code = torch.cat([short_fofe, long_fofe], dim=-1)
        return fofe_code
    
    def extra_repr(self):
        return 'alpha_l={alpha_l}, alpha_h={alpha_h}'.format(**self.__dict__)


class fofe_flex(nn.Module):
    def __init__(self, inplanes, alpha): 
        super(fofe_flex, self).__init__()
        self.alpha = Parameter(torch.ones(1)*alpha)
        self.alpha.requires_grad_(True)
        
    def forward(self, x):
        length = x.size(-2)
        #import pdb; pdb.set_trace()
        matrix = torch.pow(self.alpha,torch.linspace(length-1,0,length).cuda()).unsqueeze(0)
        fofe_code = matrix.matmul(x).squeeze(-2)
        return fofe_code

    def extra_repr(self):
        return 'inplanes={inplanes}'.format(**self.__dict__)


class fofe_flex_dual(nn.Module):
    def __init__(self, inplanes, alpha_l, alpha_h): 
        super(fofe_flex_dual, self).__init__()
        self.inplanes = inplanes
        self.alpha_l = Parameter(torch.ones(1)*alpha_l)
        self.alpha_h = Parameter(torch.ones(1)*alpha_h)
        self.alpha_l.requires_grad_(True)
        self.alpha_h.requires_grad_(True)
        
    def forward(self, x, x_mask):
        length = x.size(-2)
        matrix_l = torch.pow(self.alpha_l, x.new_tensor(torch.linspace(length-1,0,length))).unsqueeze(0)
        matrix_h = torch.pow(self.alpha_h, x.new_tensor(torch.linspace(length-1,0,length))).unsqueeze(0)
        mask_l = torch.pow(self.alpha_l, x.new_tensor(x_mask.sum(1)).mul(-1)).unsqueeze(1)
        mask_h = torch.pow(self.alpha_h, x.new_tensor(x_mask.sum(1)).mul(-1)).unsqueeze(1)
        fofe_l = matrix_l.matmul(x).squeeze(-2).mul(mask_l)
        fofe_h = matrix_h.matmul(x).squeeze(-2).mul(mask_h)
        fofe_code = torch.cat([fofe_l, fofe_h], dim=-1)
        return fofe_code

    def extra_repr(self):
        return 'inplanes={inplanes}'.format(**self.__dict__)


class fofe_flex_all(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe_flex_all, self).__init__()
        self.channels = channels
        self.alpha = Parameter(torch.ones(channels, 1)*alpha)
        self.alpha.requires_grad_(True)
        
    def forward(self, x, x_mask):
        length = x.size(-2)
        mask = torch.pow(self.alpha, x.new_tensor(x_mask.sum(1)).mul(-1)).unsqueeze(1).permute(2,0,1)
        matrix = torch.pow(self.alpha, x.new_tensor(torch.linspace(length-1,0,length))).unsqueeze(1)
        fofe_code = F.conv1d(x.transpose(-1,-2), matrix, bias=None, stride=1, padding=0, groups = self.channels)
        fofe_code = fofe_code.mul(mask)
        return fofe_code


class fofe_flex_all_conv(nn.Module):
    def __init__(self, inplanes, planes, alpha): 
        super(fofe_flex_all_conv, self).__init__()
        self.fofe = fofe_flex_all(inplanes, alpha)
        self.conv = nn.Sequential(
            nn.Conv1d(inplanes, planes, 1, 1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, x_mask):
        fofe_code = self.fofe(x, x_mask)
        out = self.conv(fofe_code)
        return out


class fofe_flex_all_filter(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_flex_all_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = Parameter(torch.ones(inplanes, 1)*alpha)
        self.alpha.requires_grad_(True)
        self.inverse = inverse

    def forward(self, x):
        if self.inverse:
            fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.range(0, self.length-1))).unsqueeze(1)
            x = F.pad(x,(0, self.length))
        else :
            fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.linspace(self.length-1, 0, self.length))).unsqueeze(1)
            x = F.pad(x,(self.length, 0))
        x = F.conv1d(x, fofe_kernel, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        
        return x


class fofe_flex_dual_filter(nn.Module):
    def __init__(self, inplanes, alpha_l=0.4, alpha_h=0.8, length=3, inverse=False):
        super(fofe_flex_dual_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha_l = Parameter(torch.ones(1)*alpha_l)
        self.alpha_h = Parameter(torch.ones(1)*alpha_h)
        self.alpha_l.requires_grad_(True)
        self.alpha_h.requires_grad_(True)
        self.inverse = inverse

    def forward(self, x):
        fofe_kernel_l = x.new_zeros(x.size(1), 1, self.length)
        fofe_kernel_h = x.new_zeros(x.size(1), 1, self.length)
        if self.inverse:
            fofe_kernel_l[:,:,]=torch.pow(self.alpha_l, x.new_tensor(torch.range(0, self.length-1)))
            fofe_kernel_h[:,:,]=torch.pow(self.alpha_h, x.new_tensor(torch.range(0, self.length-1)))
            x = F.pad(x,(0, self.length))
        else :
            fofe_kernel_l[:,:,]=torch.pow(self.alpha_l, x.new_tensor(torch.linspace(self.length-1, 0, self.length)))
            fofe_kernel_h[:,:,]=torch.pow(self.alpha_h, x.new_tensor(torch.linspace(self.length-1, 0, self.length)))
            x = F.pad(x,(self.length, 0))
        fofe_l = F.conv1d(x, fofe_kernel_l, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        fofe_h = F.conv1d(x, fofe_kernel_h, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        fofe_code = torch.cat([fofe_l, fofe_h], dim=1)

        return fofe_code

    def extra_repr(self):
        return 'inplanes={channels}, length={length}， ' \
            'inverse={inverse}'.format(**self.__dict__)


class fofe_flex_filter(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_flex_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = Parameter(torch.ones(1)*alpha)
        self.alpha.requires_grad_(True)
        self.inverse = inverse

    def forward(self, x):
        #if self.alpha == 1 or self.alpha == 0 :
        #    self.alpha = 0.9
        fofe_kernel = x.new_zeros(x.size(1), 1, self.length)
        if self.inverse:
            fofe_kernel[:,:,]=torch.pow(self.alpha, x.new_tensor(torch.range(0, self.length-1)))
            x = F.pad(x,(0, self.length))
        else :
            fofe_kernel[:,:,]=torch.pow(self.alpha, x.new_tensor(torch.linspace(self.length-1, 0, self.length)))
            x = F.pad(x,(self.length, 0))
        x = F.conv1d(x, fofe_kernel, bias=None, stride=1, 
                        padding=0, groups=self.channels)

        return x

    def extra_repr(self):
        return 'inplanes={inplanes}, length={length}， ' \
            'inverse={inverse}'.format(**self.__dict__)


class fofe_flex_dual_linear_filter(nn.Module):
    def __init__(self, inplanes, planes, alpha_l=0.4, alpha_h=0.8, length=3, inverse=False):
        super(fofe_flex_dual_linear_filter, self).__init__()
        self.fofe = fofe_flex_dual(inplanes, alpha_l, alpha_h)
        self.conv = nn.Sequential(
            nn.Conv1d(inplanes, planes, 1, 1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        fofe = self.fofe(x)
        out = self.conv(fofe)
        return out


class fofe_flex_dual_linear(nn.Module):
    def __init__(self, inplanes, planes, alpha_l=0.4, alpha_h=0.8, length=3, inverse=False):
        super(fofe_flex_dual_linear, self).__init__()
        self.fofe = fofe_flex_dual(inplanes, alpha_l, alpha_h)
        self.conv = nn.Sequential(
            nn.Conv1d(inplanes*2, planes*2, 1, 1, bias=False),
            nn.BatchNorm1d(planes*2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, x_mask):
        fofe = self.fofe(x, x_mask)
        out = self.conv(fofe.unsqueeze(-1))
        return out


class fofe_encoder(nn.Module):
    def __init__(self, emb_dim, fofe_alpha, fofe_max_length):
        super(fofe_encoder, self).__init__()
        self.forward_filter = []
        self.inverse_filter = []
        for i in range(fofe_max_length):
            self.forward_filter.append(fofe_flex_all_filter(emb_dim, fofe_alpha, i+1))
            self.inverse_filter.append(fofe_flex_all_filter(emb_dim, fofe_alpha, i+1, inverse=True))

        self.forward_filter = nn.ModuleList(self.forward_filter)
        self.inverse_filter = nn.ModuleList(self.inverse_filter)
    
    def fofe(self, x):
        forward_fofe = []
        inverse_fofe = []
        for forward_filter in self.forward_filter:
            forward_fofe.append(forward_filter(x).unsqueeze(-2))
        for inverse_filter in self.inverse_filter:
            inverse_fofe.append(inverse_filter(x).unsqueeze(-2))

        forward_fofe = torch.cat(forward_fofe, dim=-2)
        inverse_fofe = torch.cat(inverse_fofe, dim=-2)
        
        return forward_fofe, inverse_fofe

    def forward(self, x):
        forward_fofe, inverse_fofe = self.fofe(x)
        return forward_fofe, inverse_fofe


class fofe_encoder_dual(nn.Module):
    def __init__(self, emb_dim, fofe_alpha_l, fofe_alpha_h, fofe_max_length):
        super(fofe_encoder_dual, self).__init__()
        self.forward_filter = []
        self.inverse_filter = []
        for i in range(fofe_max_length):
            self.forward_filter.append(fofe_dual_filter(emb_dim, fofe_alpha_l, fofe_alpha_h, i+1))
            self.inverse_filter.append(fofe_dual_filter(emb_dim, fofe_alpha_l, fofe_alpha_h, i+1, inverse=True))

        self.forward_filter = nn.ModuleList(self.forward_filter)
        self.inverse_filter = nn.ModuleList(self.inverse_filter)
    
    def fofe(self, x):
        forward_fofe = []
        inverse_fofe = []
        for forward_filter in self.forward_filter:
            forward_fofe.append(forward_filter(x).unsqueeze(-2))
        for inverse_filter in self.inverse_filter:
            inverse_fofe.append(inverse_filter(x).unsqueeze(-2))

        forward_fofe = torch.cat(forward_fofe, dim=-2)
        inverse_fofe = torch.cat(inverse_fofe, dim=-2)
        
        return forward_fofe, inverse_fofe

    def forward(self, x):
        forward_fofe, inverse_fofe = self.fofe(x)
        return forward_fofe, inverse_fofe


class fofe_encoder_conv(fofe_encoder):
    def __init__(self, inplanes, planes, fofe_alpha, fofe_max_length):
        super(fofe_encoder_conv, self).__init__(inplanes, fofe_alpha, fofe_max_length)
        self.forward_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.inverse_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        forward_fofe, inverse_fofe = self.fofe(x)
        forward = self.forward_conv(forward_fofe)
        inverse = self.inverse_conv(inverse_fofe)

        return forward, inverse


class fofe_flex_all_encoder(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_flex_all_encoder, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = Parameter(torch.ones(inplanes, 1)*alpha)
        self.alpha.requires_grad_(True)
        self.inverse = inverse

    def forward(self, x, max_len):
        out = []
        for i in range(max_len):
            if self.inverse:
                fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.range(0, i-1))).unsqueeze(1)
                out.append(F.conv1d(F.pad(x,(0, i), fofe_kernel, bias=None, stride=1, 
                                padding=0, groups=self.channels).unsqueeze(-2))
            else:
                fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.linspace(i-1, 0, i))).unsqueeze(1)
                out.append(F.conv1d(F.pad(x,(i, 0)), fofe_kernel, bias=None, stride=1, 
                                padding=0, groups=self.channels).unsqueeze(-2))
        
        if self.inverse:
            fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.range(0, 64-1))).unsqueeze(1)
            out.append(F.conv1d(F.pad(x,(0, i), fofe_kernel, bias=None, stride=1, 
                            padding=0, groups=self.channels).unsqueeze(-2))
        else :
            fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.linspace(64-1, 0, 64))).unsqueeze(1)
            out.append(F.conv1d(F.pad(x,(64, 0)), fofe_kernel, bias=None, stride=1, 
                            padding=0, groups=self.channels).unsqueeze(-2))

        out = torch.cat(out, dim=-2)
        return out


class fofe_encoder_conv(fofe_encoder):
    def __init__(self, inplanes, planes, fofe_alpha_l, fofe_alpha_h, fofe_max_length):
        super(fofe_encoder_conv, self).__init__(inplanes, fofe_alpha_l, fofe_alpha_h, fofe_max_length)
        self.forward_conv = nn.Sequential(
            nn.Conv2d(inplanes*2, planes*2, 1, 1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(inplace=True)
        )
        self.inverse_conv = nn.Sequential(
            nn.Conv2d(inplanes*2, planes*2, 1, 1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        forward_fofe, inverse_fofe = self.fofe(x)
        forward = self.forward_conv(forward_fofe)
        inverse = self.inverse_conv(inverse_fofe)

        return forward, inverse
