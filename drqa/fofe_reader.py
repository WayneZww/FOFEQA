# Modification:
#  -change to support fofe_nn
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from .fofe_modules import fofe_conv1d, fofe_linear


class FOFEReader(nn.Module):
    def __init__(self, opt, padding_idx=0, embedding=None):
        super(FOFEReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                self.embedding.weight.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                offset = self.opt['tune_partial'] + 2

                def embedding_hook(grad, offset=offset):
                    grad[offset:] = 0
                    return grad

                self.embedding.weight.register_hook(embedding_hook)

        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)

        # Input size to FOFE_NN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['pos']:
            doc_input_size += opt['pos_size']
        if opt['ner']:
            doc_input_size += opt['ner_size']
        
        self.fofe_nn = FOFE_NN_split(opt['embedding_dim'], 
                                opt['fofe_alpha'],
                                opt['fofe_max_length'])
        
    def forward(self, doc, doc_f, doc_pos, doc_ner, doc_mask, query, query_mask):
        """Inputs:
        doc = document word indices             [batch * len_d]
        doc_f = document word features indices  [batch * len_d * nfeat]
        doc_pos = document POS tags             [batch * len_d]
        doc_ner = document entity tags          [batch * len_d]
        doc_mask = document padding mask        [batch * len_d]
        query = question word indices             [batch * len_q]
        query_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        doc_emb = self.embedding(doc)
        query_emb = self.embedding(query)
        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            doc_emb = nn.functional.dropout(doc_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
            query_emb = nn.functional.dropout(query_emb, p=self.opt['dropout_emb'],
                                           training=self.training)

        doc_input_list = [doc_emb, doc_f]
        if self.opt['pos']:
            doc_input_list.append(doc_pos)
        if self.opt['ner']:
            doc_input_list.append(doc_ner)
        doc_input = torch.cat(doc_input_list, 2)
        
        # Predict start and end positions
        start_scores, end_scores = self.fofe_nn(query_emb, query_mask, doc_emb, doc_mask)
        return start_scores, end_scores


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
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dims*4, emb_dims*4, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dims*4, emb_dims*2, 1, 1, bias=False),
            nn.ReLU(inplace=True)
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

    
class FOFE_NN_split(nn.Module):
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        
    def __init__(self, emb_dims, fofe_alpha, fofe_max_length, training=True):
        super(FOFE_NN_split, self).__init__()
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
        self.s_conv = nn.Conv2d(emb_dims*2, 1, 1, 1, bias=False)
        self.e_conv = nn.Conv2d(emb_dims*2, 1, 1, 1, bias=False)
        
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
        import pdb
        pdb.set_trace()
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

