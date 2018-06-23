import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from .fofe_modules import Attention, fofe_block, fofe_res_block, ln_conv, BiAttention
        
class FOFENet(nn.Module):
    def _make_layer(self, block, inplanes, planes, blocks, block_convs, 
            fofe_alpha=0.8, fofe_max_length=3, stride=1, dilation=1,  moduleList=False):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,kernel_size=1, 
                            stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

        layers = []
        layers.append(block(inplanes, planes, block_convs, fofe_alpha, fofe_max_length, dilation, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, block_convs, fofe_alpha, fofe_max_length, dilation))

        if moduleList :
            return nn.ModuleList(layers)
        else : 
            return nn.Sequential(*layers)
        
    def __init__(self, block, emb_dims, channels, fofe_alpha=0.8, fofe_max_length=3, 
                    att_bidirection=False, att_q2c=True, training=True):
        super(FOFENet, self).__init__()
        #self.inplanes=emb_dims
        self.att_bidirection = att_bidirection
        self.att_q2c = att_q2c

        self.dq_encoder = self._make_layer(block, emb_dims, channels, 6, 3, fofe_alpha, fofe_max_length)
        self.attention = Attention(channels, att_q2c, att_bidirection)
        self.model_encoder = self._make_layer(block, channels*4, channels*2, 2, 2, fofe_alpha, fofe_max_length, dilation=2)
        self.output_encoder = self._make_layer(block, channels*2, channels, 3, 3, fofe_alpha, fofe_max_length, moduleList=True)

        self.pointer_s = nn.Conv1d(channels*2, 1, 1, bias=False)
        self.pointer_e = nn.Conv1d(channels*2, 1, 1, bias=False)

    def out_encode(self, x):
        s_score = []
        e_score = []
        idx = 0
        for encoder in self.output_encoder:
            idx += 1
            x = encoder(x)
            if idx == 1:
                s_score.append(x)
                e_score.append(x)
            elif idx == 2:
                s_score.append(x)
            elif idx == 3:
                e_score.append(x)
    
        s_score = torch.cat(s_score, dim=1)
        e_score = torch.cat(e_score, dim=1)

         # calculate scores for begin and end point
        s_score = self.pointer_s(s_score).squeeze(-2)
        e_score = self.pointer_e(e_score).squeeze(-2)

        return (s_score, e_score)

    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        query_emb = torch.transpose(query_emb,-2,-1)
        doc_emb = torch.transpose(doc_emb,-2,-1)
        q_code = self.dq_encoder(query_emb)
        d_code = self.dq_encoder(doc_emb)
        att_code = self.attention(d_code, q_code)
        model_code = self.model_encoder(att_code)
        (s_score, e_score) = self.out_encode(model_code)
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


class FOFENet_Biatt(nn.Module):
    def _make_layer(self, block, inplanes, planes, blocks, block_convs, 
            fofe_alpha=0.8, fofe_max_length=3, stride=1, dilation=1,  moduleList=False):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,kernel_size=1, 
                            stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )
            
        layers = []
        layers.append(block(inplanes, planes, block_convs, fofe_alpha, fofe_max_length, dilation, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, block_convs, fofe_alpha, fofe_max_length, dilation))

        if moduleList :
            return nn.ModuleList(layers)
        else : 
            return nn.Sequential(*layers)
        
    def __init__(self, block, emb_dims, channels, fofe_alpha=0.8, fofe_max_length=3, 
                    att_bidirection=False, att_q2c=True, training=True):
        super(FOFENet_Biatt, self).__init__()
        self.dq_l_encoder = self._make_layer(block, emb_dims, channels, 4, 3, fofe_alpha, fofe_max_length)
        self.mid_attention = BiAttention(channels)
        self.dq_h_encoder = self._make_layer(block, channels*4, channels, 4, 3, fofe_alpha, fofe_max_length, dilation=2)
        self.out_attention = Attention(channels)
        self.model_encoder = self._make_layer(block, channels*4, channels*2, 2, 3, fofe_alpha, fofe_max_length, dilation=2)
        self.output_encoder = self._make_layer(block, channels*2, channels, 3, 3, fofe_alpha, fofe_max_length, moduleList=True)

        self.pointer_s = nn.Conv1d(channels*2, 1, 1, bias=False)
        self.pointer_e = nn.Conv1d(channels*2, 1, 1, bias=False)

    def out_encode(self, x):
        s_score = []
        e_score = []
        idx = 0
        for encoder in self.output_encoder:
            idx += 1
            x = encoder(x)
            if idx == 1:
                s_score.append(x)
                e_score.append(x)
            elif idx == 2:
                s_score.append(x)
            elif idx == 3:
                e_score.append(x)
    
        s_score = torch.cat(s_score, dim=1)
        e_score = torch.cat(e_score, dim=1)

         # calculate scores for begin and end point
        s_score = self.pointer_s(s_score).squeeze(-2)
        e_score = self.pointer_e(e_score).squeeze(-2)

        return (s_score, e_score)

    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        query_emb = torch.transpose(query_emb,-2,-1)
        doc_emb = torch.transpose(doc_emb,-2,-1)

        q_l_code = self.dq_l_encoder(query_emb)
        d_l_code = self.dq_l_encoder(doc_emb)
        d_att, q_att = self.mid_attention(d_l_code, q_l_code)

        q_h_code = self.dq_h_encoder(q_att)
        d_h_code = self.dq_h_encoder(d_att)
        att_code = self.out_attention(d_h_code, q_h_code)
        
        model_code = self.model_encoder(att_code)
        (s_score, e_score) = self.out_encode(model_code)
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


class FOFE_NN(nn.Module):
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        
    def __init__(self, emb_dims, fofe_alpha, fofe_max_length, training=True):
        super(FOFE_NN, self).__init__()
        self.doc_fofe_conv = []
        for i in range(2, fofe_max_length+1):
            self.doc_fofe_conv.append(fofe_conv1d(emb_dims, fofe_alpha, i, i))
        self.doc_fofe_conv = nn.ModuleList(self.doc_fofe_conv)
        self.query_fofe = fofe_linear(emb_dims, fofe_alpha)
        self.emb_dims = emb_dims
        self.fnn = nn.Sequential(
            nn.Conv2d(emb_dims*2, emb_dims*4, 1, 1, bias=False),
            nn.BatchNorm2d(emb_dims*4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(emb_dims*4, emb_dims*4, 1, 1, bias=False),
            nn.BatchNorm2d(emb_dims*4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(emb_dims*4, emb_dims*2, 1, 1, bias=False),
            nn.BatchNorm2d(emb_dims*2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.s_conv = nn.Conv2d(emb_dims*2, 1, ((fofe_max_length-1),1), 1, bias=False)
        self.e_conv = nn.Conv2d(emb_dims*2, 1, ((fofe_max_length-1),1), 1, bias=False)
        
        self.s_conv.apply(self.weights_init)      
        self.e_conv.apply(self.weights_init) 
        self.fnn.apply(self.weights_init) 
        self.doc_fofe_conv.apply(self.weights_init)

    def dq_fofe(self, query, document):
        query_fofe_code = self.query_fofe(query)
        q_mat = []
        for i in range(document.size(-2)):
            q_mat.append(query_fofe_code)
        query_mat = torch.transpose(torch.cat(q_mat,-2),-2,-1).unsqueeze(-2)
        fofe_out = []
        for fofe_layer in self.doc_fofe_conv:
            fofe_out.append(torch.cat([fofe_layer(document).unsqueeze(-2),query_mat],-3))
        fofe_out = torch.cat(fofe_out,-2)
        return fofe_out

    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        
        fofe_code = self.dq_fofe(query_emb, doc_emb)
        x = self.fnn(fofe_code)
        
        # calculate scores for begin and end point
        s_score = self.s_conv(x).squeeze(-2).squeeze(-2)
        e_score = self.e_conv(x).squeeze(-2).squeeze(-2)
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




    
class FOFE_NN_att(nn.Module):
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        
    def __init__(self, emb_dims, fofe_alpha, fofe_max_length, training=True):
        super(FOFE_NN_att, self).__init__()
        self.doc_fofe_conv = []
        for i in range(2, fofe_max_length+1):
            self.doc_fofe_conv.append(fofe_conv1d(emb_dims, fofe_alpha, i, i))
        self.doc_fofe_conv = nn.ModuleList(self.doc_fofe_conv)
        self.query_fofe = fofe_linear(emb_dims, fofe_alpha)
        self.emb_dims = emb_dims
        self.fofe_max_length = fofe_max_length
        self.fnn = nn.Sequential(
            nn.Conv2d(emb_dims*2, emb_dims*4, 1, 1, bias=False),
            nn.BatchNorm2d(emb_dims*4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(emb_dims*4, emb_dims*4, 1, 1, bias=False),
            nn.BatchNorm2d(emb_dims*4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(emb_dims*4, emb_dims*2, 1, 1, bias=False),
            nn.BatchNorm2d(emb_dims*2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.s_conv = nn.Conv2d(emb_dims*2, 1, ((fofe_max_length-1),1), 1, bias=False)
        self.e_conv = nn.Conv2d(emb_dims*2, 1, ((fofe_max_length-1),1), 1, bias=False)
        
        self.s_conv.apply(self.weights_init)      
        self.e_conv.apply(self.weights_init) 
        self.fnn.apply(self.weights_init) 
        self.doc_fofe_conv.apply(self.weights_init)

    def dq_fofe(self, query, document):
        query_fofe_code = self.query_fofe(query)
        q_mat = []
        
        for i in range(document.size(-2)):
            q_mat.append(query_fofe_code)
        query_mat = torch.transpose(torch.cat(q_mat,-2),-2,-1).unsqueeze(-2)
        fofe_out = []
        
        for fofe_layer in self.doc_fofe_conv:
            fofe_out.append(torch.cat([fofe_layer(document).unsqueeze(-2),query_mat],-3))
        fofe_out = torch.cat(fofe_out,-2)
        return fofe_out

    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        fofe_code = self.dq_fofe(query_emb, doc_emb)
        x = self.fnn(fofe_code)
        # calculate scores for begin and end point
        s_score = self.s_conv(x)
        e_score = self.e_conv(x)
        s_score = torch.split(s_score, 1, dim=-2)
        e_score = torch.split(e_score, 1, dim=-2)
        # mask scores
        for i in range(self.fofe_max_length-1): 
            s_score[i].squeeze_(-2).squeeze_(-2).data.masked_fill_(doc_mask.data, -float('inf'))
            e_score[i].squeeze_(-2).squeeze_(-2).data.masked_fill_(doc_mask.data, -float('inf'))
            score_s = []
            score_e = []
            
            if self.training:
                # In training we output log-softmax for NLL
                score_s.append(F.log_softmax(s_score[i], dim=1))
                score_e.append(F.log_softmax(e_score[i], dim=1))
            else:
                # ...Otherwise 0-1 probabilities
                s_score[i] = F.softmax(s_score[i], dim=1)
                e_score[i] = F.softmax(e_score[i], dim=1)
                
        return score_s, score_e




class FOFE_CNN(nn.Module):
    def __init__(self, emb_dims, fofe_alpha, fofe_max_length, training=True):
        super(FOFE_NN, self).__init__()
        self.doc_fofe_conv = []
        for i in range(3, fofe_max_length+1, 2):
            self.doc_fofe_conv.append(fofe_conv1d(emb_dims, fofe_alpha, i, i))
        self.doc_fofe_conv = nn.ModuleList(self.doc_fofe_conv)
        self.query_fofe = fofe_linear(emb_dims, fofe_alpha)
        self.emb_dims = emb_dims
        self.cnn = nn.Sequential(
            fofe_conv2d(emb_dims, fofe_alpha, 3, 1),
            fofe_conv2d(emb_dims, fofe_alpha, 3, 1),
            fofe_conv2d(emb_dims, fofe_alpha, 3, 1 ),           
        )
        self.s_conv = nn.Conv2d(emb_dims*2, 1, ((fofe_max_length-1)//2,1), 1, bias=False)
        self.e_conv = nn.Conv2d(emb_dims*2, 1, ((fofe_max_length-1)//2,1), 1, bias=False)       
        
    def dq_fofe(self, query, document):
        query_fofe_code = self.query_fofe(query)
        q_mat = []
        for i in range(document.size(-2)):
            q_mat.append(query_fofe_code)
        query_mat = torch.transpose(torch.cat(q_mat,-2),-2,-1).unsqueeze(-2)
        fofe_out = []
        for fofe_layer in self.doc_fofe_conv:
            fofe_out.append(torch.cat([fofe_layer(document).unsqueeze(-2),query_mat],-3))
        fofe_out = torch.cat(fofe_out,-2)
        print(fofe_out.shape)
        
        return fofe_out

    def forward(self, query, document):
        fofe_code = self.dq_fofe(query, document)
        x = self.cnn(fofe_code)
        s_score = self.s_conv(x).squeeze(-2).squeeze(-2)
        e_score = self.e_conv(x).squeeze(-2).squeeze(-2)
        if self.training:
            # In training we output log-softmax for NLL
            s_score = F.log_softmax(s_score, dim=1)
            e_score = F.log_softmax(e_score, dim=1)
        else:
            # ...Otherwise 0-1 probabilities
            s_score = F.softmax(s_score, dim=1)
            e_score = F.softmax(e_score, dim=1)

        return s_score, e_score


