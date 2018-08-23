# Modification:
#  -change to support fofe_nn
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import random
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
            

from .fofe_modules import fofe_multi, fofe_multi_encoder, fofe, fofe_filter, fofe_flex_all, fofe_flex_all_filter, \
                            bidirect_fofe_tricontext, bidirect_fofe, bidirect_fofe_multi_tricontext, bidirect_fofe_multi

from .fofe_net import FOFE_NN_att, FOFE_NN
from .utils import tri_num
from .focal_loss import FocalLoss1d


class FOFEReader(nn.Module):
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def __init__(self, opt, padding_idx=0, embedding=None):
        super(FOFEReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            print("pretrained embedding used")
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                self.embedding.weight.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                offset = self.opt['tune_partial'] + 2
                def embedding_hook_v1(grad, offset=offset):
                    grad[offset:] = 0
                    return grad
                self.embedding.weight.register_hook(embedding_hook_v1)
            else:
                def embedding_hook_v2(grad):
                    grad[:1] = 0
                    return grad
                self.embedding.weight.register_hook(embedding_hook_v2)

        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)

        # Input size to fofe_encoder: word_emb + manual_features
        doc_input_size = opt['embedding_dim']+opt['num_features']
        if opt['pos']:
            doc_input_size += opt['pos_size']
        if opt['ner']:
            doc_input_size += opt['ner_size']

        n_ctx_types = 1
        query_input_size = opt['embedding_dim']       
        if (self.opt['contexts_incl_cand']):
            n_ctx_types += 2
        if (self.opt['contexts_excl_cand']):
            n_ctx_types += 2
        n_fofe_direction = 2    # SINCE: doc candidate fofe and query fofe are bidectional fofe
        n_fofe_alphas = len(opt['fofe_alpha'])
        # Input size to fofe_fnn:
        fnn_input_size = (doc_input_size * (n_ctx_types+n_fofe_direction-1) + \
                          query_input_size * n_fofe_direction) * n_fofe_alphas

        # Wayne's Version FOFE Encoders
        if opt['filter'] == 'fofe':
            self.fofe_encoder = fofe_multi_encoder(fofe_filter, doc_input_size, opt['fofe_alpha'],  opt['fofe_max_length'])
            self.fofe_linear = fofe_multi(fofe, opt['embedding_dim'], opt['fofe_alpha'])
        elif opt['filter'] == 'flex_all':
            self.fofe_encoder = fofe_multi_encoder(fofe_flex_all_filter, doc_input_size, opt['fofe_alpha'],  opt['fofe_max_length'])
            self.fofe_linear = fofe_multi(fofe_flex_all, opt['embedding_dim'], opt['fofe_alpha'])

        # Sed's Version FOFE Encoders
        self.doc_fofe_tricontext_encoder = bidirect_fofe_multi_tricontext(opt['fofe_alpha'], doc_input_size, opt)
        self.query_fofe_encoder = bidirect_fofe_multi(opt['fofe_alpha'], query_input_size)

        if opt['net_arch'] == 'FNN':
            self.fnn = FOFE_NN(fnn_input_size, opt['hidden_size'])
        elif opt['net_arch'] == 'FNN_att':
            self.fnn = FOFE_NN_att(fnn_input_size, opt['hidden_size'])
        elif opt['net_arch'] == 'simple':
            self.fnn = nn.Sequential(
                nn.Linear(fnn_input_size, opt['hidden_size']*4),
                nn.BatchNorm1d( opt['hidden_size']*4),
                nn.ReLU(inplace=True),
                nn.Linear(opt['hidden_size']*4, opt['hidden_size']*4),
                nn.BatchNorm1d( opt['hidden_size']*4),
                nn.ReLU(inplace=True),
                nn.Linear(opt['hidden_size']*4, opt['hidden_size']*4),
                nn.BatchNorm1d( opt['hidden_size']*4),
                nn.ReLU(inplace=True),
#                nn.Dropout(0.1),
                nn.Linear(opt['hidden_size']*4, 2),
            )
        else:
            raise Exception('Architecture undefined!')
        
        if opt['focal_alpha'] != 0:
            self.fl_loss = FocalLoss1d(2, gamma=opt['focal_gamma'], alpha=opt['focal_alpha'])
        else:
            self.fl_loss = None
        
        self.sentence_selector = nn.Sequential(
                nn.Linear((doc_input_size+query_input_size)*2, opt['hidden_size']*4, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(opt['hidden_size']*4, opt['hidden_size']*4, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(opt['hidden_size']*4, opt['hidden_size']*4, bias=True),
                nn.ReLU(inplace=True),
#                nn.Dropout(0.1),
                nn.Linear(opt['hidden_size']*4, 2),
            )

        # TODO: DISCUSS WITH WAYNE; for now when opt['neg_ratio'] <= 0, just use default weight
        # originally: self.ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1/opt['neg_ratio']]))
        if opt['neg_ratio'] > 0:
            self.ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1/opt['neg_ratio']]), ignore_index=-1)
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        print(self)
        self.apply(self.weights_init)
        self.count=0

    def rank_cand_select(self, cands_ans_pos, scores, batch_size):
        n_cands = scores.size(0)
        assert n_cands % batch_size == 0, "Error: total n_cands should be multiple of batch_size"
        n_cands_per_batch = round(n_cands / batch_size)
        predict_s = []
        predict_e = []
        for i in range(batch_size):
            base_idx = i*n_cands_per_batch
            score, idx = scores[base_idx:base_idx+n_cands_per_batch].max(dim=0)
            _predict = cands_ans_pos[idx.item()]
            _predict_s = round(_predict[0].item())
            _predict_e = round(_predict[1].item())
            #   Round float to int, b/c _predict_s & _predict_s is an index (so it was a whole number with float type).
            predict_s.append(_predict_s)
            predict_e.append(_predict_e)
        return predict_s, predict_e
    
    def sample_two_stage(self, doc_emb, doc_mask, query_emb, query_mask, doc_pos, target_s=None, target_e=None):
        pos_tagger = doc_pos[:,:,7]
        pos_tagger[:,-1] = 1
        sentence_pos = []
        sentence_start = 0
        ans_target_s = 0
        ans_target_e = 0
        sentence_target_pos = 0
        sentence_target_len = 0
        for i in range(pos_tagger.size(-1)):
            if target_e is not None and target_e.item() == i:
                sentence_target_pos = len(sentence_pos)
                ans_target_e = target_e.item() - sentence_start
                ans_target_s = target_s.item() - sentence_start
            if pos_tagger[:,i].item() == 1:
                sentence_pos.append(i)
                sentence_start = i
        #import pdb; pdb.set_trace()
        if target_e is not None:
            sentence_target = doc_emb.new_zeros((len(sentence_pos),)).long()
            if sentence_target_pos == len(sentence_pos):
                import pdb; pdb.set_trace()
            sentence_target[sentence_target_pos] = 1
            return sentence_pos, sentence_target,  ans_target_s, ans_target_e
        else:
            return sentence_pos

    def sentence_select(self, doc_emb, query_emb, query_mask, sentence_pos):
        query_fofe = self.fofe_linear(query_emb, query_mask.float())

        query_batch = [query_fofe]
        sentence_batch = [self.fofe_linear(doc_emb[:,:sentence_pos[0]])]
        for i in range(1, len(sentence_pos)):
            sentence_batch.append(self.fofe_linear(doc_emb[:,sentence_pos[i-1]:sentence_pos[i]]))
            query_batch.append(query_fofe)
        
        query_batch = torch.cat(query_batch, dim=0)  
        sentence_batch = torch.cat(sentence_batch, dim=0) 
        sq_input = torch.cat([query_batch, sentence_batch], dim=1)
        sentence_score = self.sentence_selector(sq_input)
        sentence = F.softmax(sentence_score, dim=1)
        v, idx = torch.topk(sentence[:,1], 1)
        if idx.item() == 0:
            sentence_s = 0
            sentence_e = sentence_pos[0]
        else:
            sentence_s = sentence_pos[idx.item()-1]
            sentence_e = sentence_pos[idx.item()]

        return sentence_score, sentence_s, sentence_e


    def sample_via_sentence(self, doc_emb, query_emb, query_mask, target_s=None, target_e=None):
        doc_emb = doc_emb.transpose(-2,-1)
        doc_len = doc_emb.size(-1)
        batchsize = doc_emb.size(0)

        if target_s is not None:
            ans_span = target_e - target_s
            max_len = int(min(max(self.opt['max_len'], ans_span+1), doc_len))
            score_batch = []
            can_score = doc_emb.new_zeros((batchsize, 1, max_len, doc_len+1))
        else:
            max_len = int(min(self.opt['max_len'], doc_len))

        forward_fofe, inverse_fofe = self.fofe_encoder(doc_emb, max_len)
        
        # generate ctx and ans batch
        l_ctx_ex_batch = []
        r_ctx_ex_batch = []
        l_ctx_in_batch = []
        r_ctx_in_batch = []
        forward_ans_batch = []
        inverse_ans_batch = []
        pos_batch = []
        starts = []
        ends = []
        for i in range(max_len):
            l_ctx_ex_batch.append(forward_fofe[:, :, -1, 0:doc_len-i])
            l_ctx_in_batch.append(forward_fofe[:, :, -1, 1+i:doc_len+1])
            r_ctx_ex_batch.append(inverse_fofe[:, :, -1, 1+i:doc_len+1])
            r_ctx_in_batch.append(inverse_fofe[:, :, -1, 0:doc_len-i])
            forward_ans_batch.append(forward_fofe[:, :, i, 1+i:doc_len+1])
            inverse_ans_batch.append(inverse_fofe[:, :, i, 0:doc_len-i])
            if target_s is not None:
                for j in range(batchsize):
                    ans_e = target_e + 1
                    ans_s = target_s + 1
                    ans_len = ans_span + 1
                    if ans_len == i+1 and target_s != -1:
                        can_score[j, :, i, ans_e:ans_s+i+1].fill_(1)
                score_batch.append(can_score[:, :, i, 1+i:doc_len+1])
            
            starts.append(torch.arange(0, doc_len-i, device=doc_emb.device))
            ends.append(torch.arange(i, doc_len, device=doc_emb.device))

        l_ctx_in_batch =torch.cat(l_ctx_in_batch, dim=-1)
        l_ctx_ex_batch =torch.cat(l_ctx_ex_batch, dim=-1)
        r_ctx_ex_batch =torch.cat(r_ctx_ex_batch, dim=-1)
        r_ctx_in_batch =torch.cat(r_ctx_in_batch, dim=-1)
        inverse_ans_batch = torch.cat(inverse_ans_batch, dim=-1)
        forward_ans_batch = torch.cat(forward_ans_batch, dim=-1)
        
        # generate query batch
        query_fofe = self.fofe_linear(query_emb, query_mask.float()).unsqueeze(-1)
        query_batch = []
        for i in range(forward_ans_batch.size(-1)):
            query_batch.append(query_fofe)
        query_batch = torch.cat(query_batch, dim=-1)  
        dq_input = torch.cat([forward_ans_batch, inverse_ans_batch, l_ctx_ex_batch, l_ctx_in_batch, r_ctx_ex_batch, r_ctx_in_batch, query_batch], dim=1)

        starts = torch.cat(starts, dim=0).long().unsqueeze(-1)
        ends = torch.cat(ends, dim=0).long().unsqueeze(-1)
        cands_ans_pos = torch.cat([starts, ends], dim=-1)

        # change size
        dq_input = torch.reshape(dq_input.transpose(-1,-2), (dq_input.size(0)*dq_input.size(-1), dq_input.size(1)))

        if target_s is not None:
            target_score = torch.cat(score_batch, dim=-1).squeeze(1).long()
            #change size
            target_score = torch.reshape(target_score, (dq_input.size(0),))
            return dq_input, target_score, cands_ans_pos

        return dq_input, cands_ans_pos

    # #--------------------------------------------------------------------------------
           
    # def sample_via_conv(self, doc_emb, doc_mask, query_emb, query_mask, doc_pos, target_s=None, target_e=None):
    #     doc_emb = doc_emb.transpose(-2,-1)
    #     doc_len = doc_emb.size(-1)
    #     batchsize = doc_emb.size(0)
    #     pos_tagger = doc_pos[:,:,7]

    #     if target_s is not None:
    #         ans_span = target_e - target_s
    #         v, idx = torch.max(ans_span, dim=0)
    #         max_len = int(min(max(self.opt['max_len'], v+1), doc_len))
    #         score_batch = []
    #         can_score = doc_emb.new_zeros((batchsize, 1, max_len, doc_len+1))
    #     else:
    #         max_len = int(min(self.opt['max_len'], doc_len))

    #     forward_fofe, inverse_fofe = self.fofe_encoder(doc_emb, max_len)
        
    #     # generate ctx and ans batch
    #     l_ctx_ex_batch = []
    #     r_ctx_ex_batch = []
    #     l_ctx_in_batch = []
    #     r_ctx_in_batch = []
    #     forward_ans_batch = []
    #     inverse_ans_batch = []
    #     mask_batch = []
    #     pos_batch = []
    #     starts = []
    #     ends = []
    #     for i in range(max_len):
    #         l_ctx_ex_batch.append(forward_fofe[:, :, -1, 0:doc_len-i])
    #         l_ctx_in_batch.append(forward_fofe[:, :, -1, 1+i:doc_len+1])
    #         r_ctx_ex_batch.append(inverse_fofe[:, :, -1, 1+i:doc_len+1])
    #         r_ctx_in_batch.append(inverse_fofe[:, :, -1, 0:doc_len-i])
    #         forward_ans_batch.append(forward_fofe[:, :, i, 1+i:doc_len+1])
    #         inverse_ans_batch.append(inverse_fofe[:, :, i, 0:doc_len-i])
    #         if target_s is not None:
    #             for j in range(batchsize):
    #                 ans_e = target_e[j].item()+1
    #                 ans_s = target_s[j].item()+1
    #                 ans_len = ans_span[j].item()+1
    #                 if ans_len == i+1:
    #                     can_score[j, :, i, ans_e:ans_s+i+1].fill_(1)
    #             score_batch.append(can_score[:, :, i, 1+i:doc_len+1])
    #         # add pos tagger
    #         _pos_tagger = doc_emb.new_zeros(pos_tagger.shape, dtype=torch.long)
    #         for j in range(batchsize):
    #             for k in range(doc_len):
    #                 if pos_tagger[j, k].item() == 1 :
    #                     for t in range(i+1):
    #                         if target_e[j].item() == k + t :
    #                             continue
    #                         else:
    #                             if k+t <= doc_len-1 :
    #                                  _pos_tagger[j, k+t].fill_(1)

    #         pos_batch.append(_pos_tagger[:, i:])
    #         mask_batch.append(doc_mask[:, i:])
    #         starts.append(torch.arange(0, doc_len-i, device=doc_emb.device))
    #         ends.append(torch.arange(i, doc_len, device=doc_emb.device))

    #     l_ctx_in_batch =torch.cat(l_ctx_in_batch, dim=-1)
    #     l_ctx_ex_batch =torch.cat(l_ctx_ex_batch, dim=-1)
    #     r_ctx_ex_batch =torch.cat(r_ctx_ex_batch, dim=-1)
    #     r_ctx_in_batch =torch.cat(r_ctx_in_batch, dim=-1)
    #     inverse_ans_batch = torch.cat(inverse_ans_batch, dim=-1)
    #     forward_ans_batch = torch.cat(forward_ans_batch, dim=-1)

    #     # generate query batch
    #     query_fofe = self.fofe_linear(query_emb, query_mask.float())
    #     query_batch = []
    #     for i in range(forward_ans_batch.size(-1)):
    #         query_batch.append(query_fofe)
    #     query_batch = torch.cat(query_batch, dim=-1)  
    #     dq_input = torch.cat([forward_ans_batch, inverse_ans_batch, l_ctx_ex_batch, l_ctx_in_batch, r_ctx_ex_batch, r_ctx_in_batch, query_batch], dim=1)

    #     mask_batch = torch.cat(mask_batch, dim=-1)
    #     pos_batch = torch.cat(pos_batch, dim=-1)
    #     starts = torch.cat(starts, dim=0).long().unsqueeze(-1)
    #     ends = torch.cat(ends, dim=0).long().unsqueeze(-1)
    #     cands_ans_pos = torch.cat([starts, ends], dim=-1)

    #     # change size
    #     dq_input = torch.reshape(dq_input.transpose(-1,-2), (dq_input.size(0)*dq_input.size(-1), dq_input.size(1)))
    #     mask_batch = torch.reshape(mask_batch, (dq_input.size(0), 1))
    #     pos_batch = torch.reshape(pos_batch, (dq_input.size(0), 1))
    #     mask = mask_batch.long() + pos_batch
    #     mask = torch.gt(mask, 0)
    #     if target_s is not None:
    #         target_score = torch.cat(score_batch, dim=-1).squeeze(1).long()
    #         #change size
    #         target_score = torch.reshape(target_score, (dq_input.size(0),))
    #         target_score = target_score - mask.long().squeeze(1)
    #         if torch.eq(target_score, 1).sum() != batchsize :
    #             import pdb; pdb.set_trace()
    #         return dq_input, target_score, cands_ans_pos, mask

    #     return dq_input, cands_ans_pos, mask                

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
        doc = document word indices                 [batch * len_d]
        doc_f = document word features indices      [batch * len_d * nfeat]
        doc_pos = document POS tags                 [batch * len_d]
        doc_ner = document entity tags              [batch * len_d]
        doc_mask = document padding mask            [batch * len_d]
        query = question word indices               [batch * len_q]
        query_mask = question padding mask          [batch * len_q]
        """
        doc_emb, query_emb = self.input_embedding(doc, doc_f, doc_pos, doc_ner, query)
        if self.training and not self.opt['draw_score']:  
            sentence_pos, sentence_target,  ans_target_s, ans_target_e = self.sample_two_stage(doc_emb, doc_mask, query_emb, query_mask, doc_pos, target_s, target_e)
            sentence_score, sentence_s, sentence_e = self.sentence_select(doc_emb, doc_mask, query_emb, query_mask, sentence_pos)        
            if ans_target_s + sentence_s != target_s.item():   
                ans_target_s = -1
                ans_target_e = -1

            dq_input, ans_target, cands_ans_pos = self.sample_via_sentence(doc_emb[:,sentence_s:sentence_e], query_emb, query_mask, ans_target_s, ans_target_e)
            ans_scores = self.fnn(dq_input)

            loss = F.cross_entropy(sentence_score, sentence_target)
            loss = loss + self.ce_loss(ans_scores, ans_target)
            # if self.fl_loss is not None:
            #     loss = loss + self.fl_loss(scores, target_score)
            return loss
        elif self.opt['draw_score']:
            # Wayne's Version:
            sentence_pos, sentence_target, ans_target_s, ans_target_e = self.sample_two_stage(doc_emb, doc_mask, query_emb, query_mask, doc_pos, target_s, target_e)
            v, idx = torch.topk(sentence_target, 1)
            if idx.item() == 0:
                sentence_s = 0
                sentence_e = sentence_pos[0]
            else:
                sentence_s = sentence_pos[idx.item()-1]
                sentence_e = sentence_pos[idx.item()]        
            if ans_target_s + sentence_s != target_s.item():   
                ans_target_s = -1
                ans_target_e = -1

            dq_input, ans_target, cands_ans_pos = self.sample_via_sentence(doc_emb[:,sentence_s:sentence_e], query_emb, query_mask, ans_target_s, ans_target_e)
            predict_s, predict_e = self.rank_cand_select(cands_ans_pos, ans_target, 1)
            return predict_s, predict_e
        else:
            # Wayne's Version:
            sentence_pos = self.sample_two_stage(doc_emb, doc_mask, query_emb, query_mask, doc_pos)
            sentence_score, sentence_s, sentence_e = self.sentence_select(doc_emb, query_emb, query_mask, sentence_pos)        
            dq_input, cands_ans_pos = self.sample_via_sentence(doc_emb[:,sentence_s:sentence_e], query_emb, query_mask)
            ans_scores = self.fnn(dq_input)
            scores = F.softmax(ans_scores, dim=1)
            scores = scores[:,1:]
            predict_s, predict_e = self.rank_cand_select(cands_ans_pos, scores, 1)
            predict_s[0] = predict_s[0] + sentence_s
            predict_e[0] = predict_e[0] + sentence_s
            return predict_s, predict_e

            





