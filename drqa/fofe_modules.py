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
            nn.Conv1d(self.channels,self.channels*3,3,1,padding=length,
                        dilation=dilation, groups=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def _init_filter(self, channels, alpha, length, inverse):
        if not inverse :
            self.fofe_filter[:,:,].data = torch.pow(self.alpha,torch.linspace(length-1,0,length))
        else :
            self.fofe_filter[:,:,].data = torch.pow(self.alpha,torch.range(0,length-1))

    def forward(self, x): 
        x = F.conv1d(torch.transpose(x,-2,-1), self.fofe_filter, bias=None, stride=1, 
                        padding=self.padding, groups=self.channels)
        x = self.dilated_conv(x)

        return x


class fofe_conv2d(nn.Module):
    def __init__(self, emb_dims, alpha=0.9, length=1, dilation=1, inverse=False):
        super(fofe_conv1d, self).__init__()
        self.alpha = alpha
        self.length = length
        self.channels = emb_dims
        self.fofe_filter = Parameter(torch.Tensor(emb_dims,1,1,length))
        self.fofe_filter.requires_grad_(False)
        self._init_filter(emb_dims, alpha, length, inverse)
        self.padding = (length - 1)//2
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(self.channels,self.channels,(1,3),1,padding=(0,length),
                        dilation=(1,dilation), groups=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def _init_filter(self, channels, alpha, length, inverse):
        if not inverse :
            self.fofe_filter[:,:,:,].data = torch.pow(self.alpha,torch.linspace(length-1,0,length))
        else :
            self.fofe_filter[:,:,:,].data = torch.pow(self.alpha,torch.range(0,length-1))

    def forward(self, x): 
        x = F.conv2d(x, self.fofe_filter, bias=None, stride=1, 
                    padding=self.padding, groups=self.channels)
        x = self.dilated_conv(x)

        return x

    
class fofe_linear(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe_linear, self).__init__()
        self.alpha = alpha
        self.channels = channels
        self.matrix = Parameter(torch.Tensor())
        self.matrix.requires_grad_(False)
        self.linear = nn.Sequential(
            nn.Linear(channels,channels, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        length = x.size(-2)
        self.matrix.resize_(x.size(0),1,length)
        self.matrix[:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        fofe_code = torch.bmm(self.matrix,x)
        output = self.linear(fofe_code)
        return output
