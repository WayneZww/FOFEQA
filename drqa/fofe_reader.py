# Modification:
#  -change to support fofe_nn
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from .fofe_modules import fofe_conv1d, fofe, fofe_block, fofe_res_block, fofe_encoder
from .fofe_net import FOFENet, FOFE_NN


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
        
        self.fofe_encoder = fofe_encoder(doc_input_size, opt['fofe_alpha'], opt['fofe_max_length'])
        self.fofe_linear = fofe(opt['embedding_dim'], opt['fofe_alpha'])
        self.fnn = nn.Sequential(
            nn.Linear(doc_input_size*3+opt['embedding_dim'], opt['embedding_dim']*4, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(opt['embedding_dim']*4, opt['embedding_dim']*4, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(opt['embedding_dim']*4, opt['embedding_dim']*4, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(opt['embedding_dim']*4, opt['embedding_dim']*2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(opt['embedding_dim']*2, 1, bias=False),
            nn.Sigmoid()
        )

    def sample(self, doc_emb, query_emb, target_s, target_e):
        doc_emb = torch.transpose(doc_emb,-2,-1)
        forward_fofe, inverse_fofe = self.fofe_encoder(doc_emb)
        ans_span = target_e - target_s
        ans_idx = doc_emb.new_zeros(self.opt['sample_num'],dtype=torch.long)
        ans = torch.index_select(forward_fofe[:,:, ans_span.item(), target_e.item()], dim=0, index=ans_idx)
        left_pos = torch.randint(self.opt['sample_min_len'], self.opt['fofe_max_length'], ans_idx.shape, 
                                    dtype=torch.long, device=doc_emb.device)
        right_pos = torch.randint(self.opt['sample_min_len'], self.opt['fofe_max_length'], ans_idx.shape, 
                                    dtype=torch.long, device=doc_emb.device)
        query_fofe = self.fofe_linear(query_emb)
        query_batch = []
        for i in range(self.opt['sample_num']):
            query_batch.append(query_fofe)
        query_batch = torch.cat(query_batch, dim=0)
        left_ctx = torch.index_select(forward_fofe[:,:,:,target_s.item()], dim=-1, index=left_pos).squeeze(0).transpose(-1,-2)
        right_ctx = torch.index_select(inverse_fofe[:,:,:,target_e.item()], dim=-1, index=right_pos).squeeze(0).transpose(-1,-2)
        dq_input = torch.cat([left_ctx, ans, right_ctx, query_batch], dim=-1)
        target_score = doc_emb.new_ones(dq_input.size(0)).unsqueeze(-1)

        return dq_input, target_score
    
    def scan_all(self, doc_emb, doc_mask):
        forward_fofe, inverse_fofe = self.fofe_encoder(doc_emb)

        doc_input = doc_emb

        return doc_input

    def input_embedding(self, doc, doc_f, doc_pos, doc_ner, query):
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

        return doc_input, query_emb
        
    def forward(self, doc, doc_f, doc_pos, doc_ner, doc_mask, query, query_mask, target_s, target_e):
        """Inputs:
        doc = document word indices             [batch * len_d]
        doc_f = document word features indices  [batch * len_d * nfeat]
        doc_pos = document POS tags             [batch * len_d]
        doc_ner = document entity tags          [batch * len_d]
        doc_mask = document padding mask        [batch * len_d]
        query = question word indices             [batch * len_q]
        query_mask = question padding mask        [batch * len_q]
        """
        doc_emb, query_emb = self.input_embedding(doc, doc_f, doc_pos, doc_ner, query)
        
        if self.training :
            dq_input, target_score = self.sample(doc_emb, query_emb, target_s, target_e)
            score = self.fnn(dq_input)
            loss = F.mse_loss(score, target_score)
            return loss
        else :
            #TODO: scan all possibilities and rank to choose the best match
            nn_input = scan_all()
            score = self.fnn(query_emb, query_mask, doc_emb, doc_mask)
            score_s, score_e = self.rank_select(nn_input, doc_emb, doc_mask)
            return score_s, score_e 


