import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class fofe_conv1d(nn.Module):
    def __init__(self, emb_dims, alpha=0.9, length=1, inverse=False):
        super(fofe_conv1d, self).__init__()
        self.alpha = alpha
        self.length = length
        self.channels = emb_dims
        self.fofe_filter = self._init_filter(emb_dims, alpha, length, inverse)
        self.padding = (length - 1)//2
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(self.channels,self.channels*3,3,1,padding=length,
                        dilation=length, groups=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def _init_filter(self, channels, alpha, length, inverse):
        fofe_filter = torch.Tensor(channels,1,length)
        if not inverse :
            fofe_filter[:,:,]=torch.pow(self.alpha,torch.linspace(length-1,0,length))
        else :
            fofe_filter[:,:,]=torch.pow(self.alpha,torch.range(0,length-1))
        return fofe_filter

    def forward(self, x): 
        x = F.conv1d(torch.transpose(x,-2,-1), self.fofe_filter, bias=None, stride=1, 
                        padding=self.padding, groups=self.channels)
        x = self.dilated_conv(x)

        return x

    
class fofe_linear(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe_linear, self).__init__()
        self.alpha = alpha
        self.channels = channels
        self.linear = nn.Sequential(
            nn.Linear(channels,channels, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        length = x.size(-2)
        matrix = torch.Tensor(x.size(0),1,length)
        matrix[:,]=torch.pow(self.alpha,torch.linspace(length-1,0,length))
        fofe_code = torch.bmm(matrix,x)
        output = self.linear(fofe_code)
        return output