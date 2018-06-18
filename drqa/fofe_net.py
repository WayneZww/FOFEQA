import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from .fofe_modules import Attention, fofe_block, fofe_res_block
        
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
        (s_score, e_score) = self.out_encode(x)
        
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

