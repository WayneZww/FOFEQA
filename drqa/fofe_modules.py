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


class fofe_block(nn.Module):
    def __init__(self, inplanes, planes, alpha=0.8, length=3, dilation=3, inverse=False):
        super(fofe_block, self).__init__()
        self.length = length
        self.channels = inplanes
        self.fofe_filter = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_filter.requires_grad_(False)
        self._init_filter(alpha, length, inverse)
        self.padding = (length - 1)//2
        self.conv = nn.Sequential(nn.Conv1d(inplanes, planes,3,1,padding=length,
                        dilation=dilation, groups=1, bias=False),
                    nn.BatchNorm1d(planes),
                    nn.LeakyReLU(0.1, inplace=True))

    def _init_filter(self, alpha, length, inverse):
        if not inverse :
            self.fofe_filter[:,:,].copy_(torch.pow(alpha,torch.linspace(length-1,0,length)))
        else :
            self.fofe_filter[:,:,].copy_(torch.pow(alpha,torch.range(0,length-1)))

    def forward(self, x): 
        x = torch.transpose(x,-2,-1)
        if (self.length % 2 == 0) :
            x = F.pad(x, (0,1), mode='constant', value=0)  
        x = F.conv1d(x, self.fofe_filter, bias=None, stride=1, 
                        padding=self.padding, groups=self.channels)
        x = self.conv(x)
    
        return x
    

class fofe_res_block(nn.Module):
    def __init__(self, inplanes, planes, convs=3, alpha=0.7, length=3, dilation=3, downsample=None, inverse=False):
        super(fofe_res_block, self).__init__()
        self.length = length
        self.channels = inplanes
        self.fofe_filter = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_filter.requires_grad_(False)
        self._init_filter(alpha, length, inverse)
        self.padding = (length - 1)//2
        
        self.conv = []
        self.conv.append(nn.Conv1d(inplanes, planes,3,1,padding=length,
                        dilation=dilation, groups=1, bias=False))
        self.conv.append(nn.BatchNorm1d(planes))
        for i in range(1, convs):
            self.conv.append(nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                nn.Conv1d(planes, planes, 3, 1, 1, 1, bias=False),
                nn.BatchNorm1d(planes)
            ))
        self.conv = nn.Sequential(*self.conv)
    
        self.relu = nn.LeakyReLU(0.1, inplace=True) 
        self.downsample = downsample

    def _init_filter(self, alpha, length, inverse):
        if not inverse :
            self.fofe_filter[:,:,].copy_(torch.pow(alpha,torch.linspace(length-1,0,length)))
        else :
            self.fofe_filter[:,:,].copy_(torch.pow(alpha,torch.range(0,length-1)))

    def forward(self, x): 
        x = torch.transpose(x,-2,-1)
        if (self.length % 2 == 0) :
            x = F.pad(x, (0,1), mode='constant', value=0)
        
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out = F.conv1d(x, self.fofe_filter, bias=None, stride=1, 
                        padding=self.padding, groups=self.channels)
        out = self.conv(out)
        out = self.bn(out)
        
        out += residual
        out = self.relu(out)
        return x


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


class Simility(nn.Module):
    def __init__(self, planes):
        super(Simility, self).__init__():
        self.W = nn.Conv2d(planes*3, planes, 1, 1, bias=False)
        self.W.weight.data = nn.init.kaiming_normal_(self.W.weight.data)

    def forward(self, doc, query):
        d_length = doc.size(-1)
        q_length = query.size(-1)
        d_matrix = []
        q_matrix = []
        a_matrix = []
        for i in range(q_length):
            d_matrix.append(doc)
        for j in range(d_length):
            q_matrix.append(query)
        d_matrix = torch.cat(d_matrix, dim=-1)
        q_matrix = torch.cat(q_matrix, dim=-1)
        s_matrix = d_matrix.bmm(q_matrix)
        q_matrix = q_matrix.transpose(-1,-2)

        a_matrix.append(d_matrix)
        a_matrix.append(q_matrix)
        a_matrix.append(s_matrix)
        a_matrix = torch.cat(a_matrix, dim=1)
        simility = self.W(a_matrix)
        

        return simility


class Attention(nn.Module):
    def __init__(self, planes, q2c=True, bidirection=False):
        super(Attention, self).__init__():
        self.simility = Simility(planes)
        self.q2c = q2c
        self.bidirection = bidirection
    
    def forward(self, doc, query):
        simility = self.simility(doc, query)
        if self.bidirection :
            q2c_att = F.softmax(simility, dim=-1)
            c2q_att = F.softmax(simility, dim=-2)
            return (q2c_att, c2q_att)
        elif self.q2c:
            q2c_att = F.softmax(simility, dim=-1)
            return q2c_att
        else :
            c2q_att = F.softmax(simility, dim=-2)
            return c2q_att


        
class FOFENet(nn.Module):
    def _make_layer(self, block, planes, blocks, block_convs, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, block_convs, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_convs))

        return nn.Sequential(*layers)
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        
    def __init__(self, block, emb_dims, fofe_alpha, fofe_max_length, att_bidirection=False, att_q2c=True, training=True):
        super(FOFENet, self).__init__()
        self.inplanes=256
        self.att_bidirection = att_bidirection
        self.att_q2c = att_q2c

        self.doc_fofe = self._make_layer(block, emb_dims, 6, 3)
        self.query_fofe = self._make_layer(block, emb_dims, 3, 1)
        self.attention = Attention(emb_dims, att_q2c, att_bidirection)
        self.output_encoder = []
        for i in range(3):
            self.output_encoder.append(block(self.inplanes, planes, 4))
        self.output_encoder = nn.ModuleList(self.output_encoder)
        self.pointer_s = nn.Conv1d(emb_dims*2, 1, 3, 1, 1, 1, bias=False)
        self.pointer_e = nn.Conv1d(emb_dims*2, 1, 3, 1, 1, 1, bias=False)

        #initial weight
        self.query_fofe.apply(self.weights_init) 
        self.doc_fofe.apply(self.weights_init)
        self.pointer_s.apply(self.weights_init) 
        self.pointer_e.apply(self.weights_init) 

    def out_encode(self, d_code):
        s_score = []
        e_score = []
        idx = 0
        for encoder in self.output_encoder:
            idx += 1
            out = encoder(d_code)
            if idx == 1:
                s_score.append(out)
                e_score.append(out)
            elif idx == 2:
                s_score.append(out)
            elif idx == 3:
                e_score.append(out)

        s_score = torch.cat(s_score, dim=1)
        e_score = torch.cat(e_score, dim=1)

         # calculate scores for begin and end point
        s_score = self.pointer_s(s_score)
        e_score = self.pointer_e(e_score)

        return (s_score, e_score)


    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        q_code = self.query_fofe(query_emb)
        d_code = self.doc_fofe(doc_emb)

        attention = self.attention(d_code, q_code)
        if self.bidirection :
            (q2c_att, c2q_att) = attention
        elif self.q2c:
            q2c_att = attention
        else :
            c2q_att = attention

        x = d_code.bmm(q2c_att)
        (s_score, e_score) = out_encode(x)
        
        # mask scores
        s_score.data.masked_fill_(doc_mask.data, -float('inf'))
        e_score.data.masked_fill_(doc_mask.data, -float('inf'))
        
        if self.training:
            # In training we output log-softmax for NLL
            s_score = F.log_softmax(s_score, dim=1)
            e_score = F.log_softmax(e_score, dim=1)
        else:
            # ...Otherwise 0-1 probabilities
            s_score = F.softmax(s_score, dim=1)
            e_score = F.softmax(e_score, dim=1)

        return s_score, e_score

