import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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
            self.inter_channels = in_channels 
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
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

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


class reg_loss(nn.Module):
    def __init__(self):
        super(reg_loss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, log_soft_x, target,  sig):
        shape = log_soft_x.shape
        sigma = target.new_full((1,), sig, dtype=torch.float)
        distribution = target.new_zeros(shape, dtype=torch.float)
        distribution.copy_(torch.range(0,shape[1]-1))
        exponent = torch.sub(distribution, target.unsqueeze(-1).float()).pow(2).mul(-1).div(2*sigma.pow(2))
        gaussian = torch.exp(exponent)
        
        if sig < 1:
            eff = sigma*torch.sqrt(torch.Tensor([2*math.pi]))
            gaussian = gaussian/eff
        loss = self.mse_loss(torch.exp(log_soft_x), gaussian)
        return loss
