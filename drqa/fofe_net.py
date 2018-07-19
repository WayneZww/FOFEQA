import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Attention, BiAttention, ASPP, SelfAttention


class FOFENet(nn.Module):
    def _make_layer(self,
                    block,
                    inplanes,
                    planes,
                    blocks,
                    block_convs,
                    fofe_alpha=0.8,
                    fofe_max_length=3,
                    stride=1,
                    dilation=1,
                    moduleList=False):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(
                    inplanes, planes, kernel_size=1, stride=stride,
                    bias=False), nn.BatchNorm1d(planes))

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                block_convs,
                fofe_alpha,
                fofe_max_length,
                dilation,
                downsample=downsample))
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, block_convs, fofe_alpha, fofe_max_length,
                      dilation))

        if moduleList:
            return nn.ModuleList(layers)
        else:
            return nn.Sequential(*layers)

    def __init__(self,
                 block,
                 emb_dims,
                 channels,
                 fofe_alpha=0.8,
                 fofe_max_length=3,
                 training=True):
        super(FOFENet, self).__init__()
        self.dq_l_encoder = self._make_layer(block, emb_dims, channels, 4, 2,
                                             fofe_alpha, fofe_max_length)
        self.dq_h_encoder = self._make_layer(
            block,
            channels,
            channels,
            2,
            2,
            fofe_alpha,
            fofe_max_length,
            dilation=2)

        self.out_attention = Attention(channels)
        self.model_encoder = self._make_layer(
            block,
            channels * 4,
            channels * 2,
            2,
            2,
            fofe_alpha,
            fofe_max_length,
            dilation=2)
        self.output_encoder = self._make_layer(
            block,
            channels * 2,
            channels,
            3,
            2,
            fofe_alpha,
            fofe_max_length,
            moduleList=True)

        self.pointer_s = nn.Conv1d(channels * 2, 1, 1, bias=False)
        self.pointer_e = nn.Conv1d(channels * 2, 1, 1, bias=False)

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

        return s_score, e_score

    def l_encode(self, query_emb, doc_emb):
        query_emb = torch.transpose(query_emb, -2, -1)
        doc_emb = torch.transpose(doc_emb, -2, -1)
        q_l_code = self.dq_l_encoder(query_emb)
        d_l_code = self.dq_l_encoder(doc_emb)
        return q_l_code, d_l_code

    def h_encode(self, q_l_code, d_l_code):
        q_h_code = self.dq_h_encoder(q_l_code)
        d_h_code = self.dq_h_encoder(d_l_code)
        att_code = self.out_attention(d_h_code, q_h_code)
        model_code = self.model_encoder(att_code)
        return model_code

    def output(self, s_score, e_score, doc_mask):
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

    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        q_l_code, d_l_code = self.l_encode(query_emb, doc_emb)
        model_code = self.h_encode(q_l_code, d_l_code)
        s_score, e_score = self.out_encode(model_code)
        s_score, e_score = self.output(s_score, e_score, doc_mask)

        return s_score, e_score


class FOFENet_Biatt(FOFENet):
    def __init__(self,
                 block,
                 emb_dims,
                 channels,
                 fofe_alpha=0.8,
                 fofe_max_length=3,
                 training=True):
        super(FOFENet_Biatt, self).__init__(
            block,
            emb_dims,
            channels,
            fofe_alpha,
            fofe_max_length,
            training=True)
        self.mid_attention = BiAttention(channels)
        self.dq_h_encoder = self._make_layer(
            block,
            channels * 4,
            channels,
            2,
            3,
            fofe_alpha,
            fofe_max_length,
            dilation=2)

    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        q_l_code, d_l_code = self.l_encode(query_emb, doc_emb)
        d_att, q_att = self.mid_attention(d_l_code, q_l_code)
        model_code = self.h_encode(q_att, d_att)
        s_score, e_score = self.out_encode(model_code)
        s_score, e_score = self.output(s_score, e_score, doc_mask)

        return s_score, e_score


class FOFENet_Biatt_ASPP(FOFENet_Biatt):
    def __init__(self,
                 block,
                 emb_dims,
                 channels,
                 fofe_alpha=0.8,
                 fofe_max_length=3,
                 training=True):
        super(FOFENet_Biatt_ASPP, self).__init__(
            block, emb_dims, channels, fofe_alpha, fofe_max_length, training)
        self.output_encoder = self._make_layer(
            block,
            channels * 4,
            channels,
            3,
            3,
            fofe_alpha,
            fofe_max_length,
            moduleList=True)
        self.aspp = ASPP(channels * 2, [1, 4, 8, 12])

    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        q_l_code, d_l_code = self.l_encode(query_emb, doc_emb)
        d_att, q_att = self.mid_attention(d_l_code, q_l_code)
        model_code = self.h_encode(q_att, d_att)
        aspp_code = self.aspp(model_code)
        #s_score, e_score = self.out_encode(aspp_code)
        s_score, e_score = self.out_encode(
            torch.cat([model_code, aspp_code], dim=1))
        s_score, e_score = self.output(s_score, e_score, doc_mask)

        return s_score, e_score


class FOFENet_Biatt_Selfatt_ASPP(FOFENet_Biatt_ASPP):
    def __init__(self,
                 block,
                 emb_dims,
                 channels,
                 fofe_alpha=0.8,
                 fofe_max_length=3,
                 training=True):
        super(FOFENet_Biatt_Selfatt_ASPP, self).__init__(
            block, emb_dims, channels, fofe_alpha, fofe_max_length, training)
        self.self_attention = SelfAttention(channels * 2)

    def forward(self, query_emb, query_mask, doc_emb, doc_mask):
        q_l_code, d_l_code = self.l_encode(query_emb, doc_emb)
        d_att, q_att = self.mid_attention(d_l_code, q_l_code)
        model_code = self.h_encode(q_att, d_att)
        att_code = self.self_attention(model_code)
        aspp_code = self.aspp(att_code)
        s_score, e_score = self.out_encode(
            torch.cat([model_code, aspp_code], dim=1))
        s_score, e_score = self.output(s_score, e_score, doc_mask)

        return s_score, e_score