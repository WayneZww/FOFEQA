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
            x = F.pad(x,(0, self.length-1))
        else :
            x = F.pad(x,(self.length-1, 0))
        x = F.conv1d(x, self.fofe_kernel, bias=None, stride=1, 
                        padding=0, groups=self.channels)

        return x


class fofe_encoder(nn.Module):
    def __init__(self, emb_dim, fofe_alpha, fofe_max_length):
        super(fofe_encoder, self).__init__()
        self.forward_filter = []
        self.inverse_filter = []
        for i in range(fofe_max_length):
            self.forward_filter.append(fofe_filter(emb_dim, fofe_alpha, i+1))
            self.inverse_filter.append(fofe_filter(emb_dim, fofe_alpha, i+1, inverse=True))

        self.forward_filter = nn.ModuleList(self.forward_filter)
        self.inverse_filter = nn.ModuleList(self.inverse_filter)

    def forward(self, x):
        forward_fofe = []
        inverse_fofe = []
        for forward_filter in self.forward_filter:
            forward_fofe.append(forward_filter(x).unsqueeze(-2))
        for inverse_filter in self.inverse_filter:
            inverse_fofe.append(inverse_filter(x).unsqueeze(-2))

        forward_fofe = torch.cat(forward_fofe, dim=-2)
        inverse_fofe = torch.cat(inverse_fofe, dim=-2)

        return forward_fofe, inverse_fofe


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
    def __init__(self, inplanes, planes, convs=3, fofe_alpha=0.9, fofe_length=3, fofe_dilation=3, downsample=None, fofe_inverse=False):
        super(fofe_res_block, self).__init__()
        self.fofe_filter = fofe_filter(inplanes, fofe_alpha, fofe_length, fofe_inverse)
        
        self.conv = []
        self.conv.append(nn.Sequential(
                            nn.Conv1d(inplanes, planes,3,1,padding=fofe_length,
                                dilation=fofe_length, groups=1, bias=False),
                            nn.BatchNorm1d(planes)))

        for i in range(1, convs):
            self.conv.append(nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                nn.Conv1d(planes, planes, 3, 1, 1, 1, bias=False),
                                nn.BatchNorm1d(planes)))

        self.conv = nn.Sequential(*self.conv)
        self.relu = nn.LeakyReLU(0.1, inplace=True) 
        self.downsample = downsample

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
        fofe_code = torch.bmm(matrix,x).squeeze(-2)
        return fofe_code


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
    def __init__(self, planes, q2c=True, bidirection=False):
        super(Attention, self).__init__()
        self.simility = Simility(planes)
        self.q2c = q2c
        self.bidirection = bidirection
    
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



