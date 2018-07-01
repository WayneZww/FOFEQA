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
        """
        self.fnn = nn.Sequential(
            nn.Linear(doc_input_size*3+opt['embedding_dim'], opt['hidden_size']*4, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(opt['hidden_size']*4, opt['hidden_size']*4, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(opt['hidden_size']*4, opt['hidden_size']*4, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(opt['hidden_size']*4, opt['hidden_size']*2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(opt['hidden_size']*2, 1, bias=False),
            nn.Sigmoid()
        )"""
        self.cnn = nn.Sequential(
            nn.Conv1d(doc_input_size*3+opt['embedding_dim'], opt['hidden_size']*4, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*4, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*4, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*2, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*2, 1, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def sample(self, doc_emb, query_emb, target_s, target_e):
        doc_emb = torch.transpose(doc_emb,-2,-1)
        forward_fofe, inverse_fofe = self.fofe_encoder(doc_emb)
        
        # generate positive ans and ctx batch
        ans_span = target_e - target_s
        positive_num = int(self.opt['sample_num']*(1-self.opt['neg_ratio']))
        batchsize = doc_emb.size(0)
        ans_minibatch = []
        l_ctx_minibatch = []
        r_ctx_minibatch = []
        for j in range(batchsize):
            ans_minibatch.append(forward_fofe[j, :, ans_span[j].item(), target_e[j].item()].unsqueeze(0))
            l_ctx_minibatch.append(forward_fofe[j, :, -1, max(target_s[j].item()-1, 0)].unsqueeze(0))
            r_ctx_minibatch.append(inverse_fofe[j, :, -1, min(target_e[j].item()+1, doc_emb.size(-1)-1)].unsqueeze(0))
        ans_minibatch = torch.cat(ans_minibatch, dim=0)
        l_ctx_minibatch = torch.cat(l_ctx_minibatch, dim=0)
        r_ctx_minibatch = torch.cat(r_ctx_minibatch, dim=0)
        positive_ctx_ans_minibatch = torch.cat([l_ctx_minibatch, ans_minibatch, r_ctx_minibatch], dim=-1).unsqueeze(-1)
        positive_ctx_ans = []
        for i in range(positive_num):                 
            positive_ctx_ans.append(positive_ctx_ans_minibatch)
        positive_ctx_ans = torch.cat(positive_ctx_ans, dim=-1)
        positive_score = doc_emb.new_ones((positive_ctx_ans.size(0), 1, positive_ctx_ans.size(-1)))
        
        # generate negative ans and ctx batch
        rand_ans = []
        rand_l_ctx = []
        rand_r_ctx = []
        negtive_num = self.opt['sample_num']-positive_num
        rand_length = torch.randperm(min(self.opt['max_len'], doc_emb.size(-1)-2))
        pos_num = negtive_num//rand_length.size(0)+1
        for i in range(rand_length.size(0)):
            rand_ans_length = rand_length[i].item()
            rand_position = torch.randint(1, doc_emb.size(-1)-rand_ans_length-1, (pos_num,), dtype=torch.long, device=doc_emb.device)
            rand_l_position = rand_position - 1
            rand_r_position = rand_position + rand_ans_length + 1
            rand_ans.append(torch.index_select(forward_fofe[:, :, rand_ans_length, :], dim=-1, index=rand_position))
            rand_l_ctx.append(torch.index_select(forward_fofe[:, :, -1, :], dim=-1, index=rand_l_position))
            rand_r_ctx.append(torch.index_select(inverse_fofe[:, :, -1, :], dim=-1, index=rand_r_position))
        
        neg_ans = torch.cat(rand_ans, dim=-1)
        neg_l_ctx = torch.cat(rand_l_ctx, dim=-1)
        neg_r_ctx = torch.cat(rand_r_ctx, dim=-1)
        rand_ctx_ans = torch.cat([neg_l_ctx, neg_ans, neg_r_ctx], dim=1)
        rand_idx = torch.randint(0, rand_ctx_ans.size(0), (negtive_num,), dtype=torch.long, device=doc_emb.device)
        negtive_ctx_ans = torch.index_select(rand_ctx_ans, dim=-1, index=rand_idx)
        negtive_score = doc_emb.new_zeros((negtive_ctx_ans.size(0), 1, negtive_ctx_ans.size(-1)))
        
        # generate query batch
        query_fofe = self.fofe_linear(query_emb).unsqueeze(-1)
        query_batch = []
        length = self.opt['sample_num']
        for i in range(length):
            query_batch.append(query_fofe)
        query_batch = torch.cat(query_batch, dim=-1)
        
        # generate net input and target with shuffle
        idx = doc_emb.new_zeros(length, dtype=torch.long)
        idx.copy_(torch.randperm(length).long())
        ctx_ans = torch.index_select(torch.cat([positive_ctx_ans, negtive_ctx_ans], dim=-1), dim=-1, index=idx)
        target_score =torch.index_select(torch.cat([positive_score, negtive_score], dim=-1), dim=-1, index=idx)
        dq_input = torch.cat([ctx_ans, query_batch], dim=1)
        
        return dq_input, target_score
    
    def scan_all(self, doc_emb, query_emb, doc_mask):
        doc_emb = torch.transpose(doc_emb,-2,-1)
        forward_fofe, inverse_fofe = self.fofe_encoder(doc_emb)

        # generate ctx and ans batch
        l_ctx_batch = []
        r_ctx_batch = []
        ans_batch = []
        mask_batch = []
        starts = []
        ends = []
        length = doc_emb.size(-1)
        batchsize = doc_emb.size(0)
        for i in range(self.opt['max_len']):
            l_ctx_batch.append(forward_fofe[:, :, -1, 0:length-2-i])
            r_ctx_batch.append(inverse_fofe[:, :, -1, i+2:length])
            ans_batch.append(forward_fofe[:, :, i, 1+i:length-1])
            mask_batch.append(doc_mask[:, 1:length-1-i])
            starts.append(torch.arange(1, length-1-i, device=doc_emb.device))
            ends.append(torch.arange(i+1, length-1, device=doc_emb.device))
        
        l_ctx_batch =torch.cat(l_ctx_batch, dim=-1)
        r_ctx_batch =torch.cat(r_ctx_batch, dim=-1)
        ans_batch = torch.cat(ans_batch, dim=-1)
        mask_batch = torch.cat(mask_batch, dim=-1)
        ctx_ans = torch.cat([l_ctx_batch, ans_batch, r_ctx_batch], dim=1)

        # generate query batch
        query_fofe = self.fofe_linear(query_emb).unsqueeze(-1)
        query_batch = []
        for i in range(ctx_ans.size(-1)):
            query_batch.append(query_fofe)
        query_batch = torch.cat(query_batch, dim=-1)

        dq_input = torch.cat([ctx_ans, query_batch], dim=1)
        starts = torch.cat(starts, dim=0).long()
        ends = torch.cat(ends, dim=0).long()

        return dq_input, starts, ends, mask_batch
    
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
        
    def forward(self, doc, doc_f, doc_pos, doc_ner, doc_mask, query, query_mask, target_s=None, target_e=None):
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
            score = self.cnn(dq_input)
            loss = F.mse_loss(score, target_score, size_average=False)
            return loss
        else :
            dq_input, starts, ends, d_mask = self.scan_all(doc_emb, query_emb, doc_mask)
            scores = self.cnn(dq_input).squeeze(1)
            scores.data.masked_fill_(d_mask.data, -float('inf'))
            v, idx = torch.max(scores, dim=-1)
            #mask = v.ge(0.5).long()
            #position = idx + 1
            position = idx
            #position = position.mul(mask) - 1
            s_idxs = torch.index_select(starts, dim=0, index=position)
            e_idxs = torch.index_select(ends, dim=0, index=position)

            return s_idxs, e_idxs 

