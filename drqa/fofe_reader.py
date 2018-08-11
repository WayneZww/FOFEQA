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
        doc_len_limit = 809
        self.doc_fofe_tricontext_encoder = bidirect_fofe_multi_tricontext(opt['fofe_alpha'],
                                                                          doc_input_size,
                                                                          opt['max_len'],
                                                                          doc_len_limit,
                                                                          opt['contexts_incl_cand'],
                                                                          opt['contexts_excl_cand'])
        self.query_fofe_encoder = bidirect_fofe_multi(opt['fofe_alpha'], query_input_size)

        if opt['net_arch'] == 'FNN':
            self.fnn = FOFE_NN(fnn_input_size, opt['hidden_size'])
        elif opt['net_arch'] == 'FNN_att':
            self.fnn = FOFE_NN_att(fnn_input_size, opt['hidden_size'])
        elif opt['net_arch'] == 'simple':
            self.fnn = nn.Sequential(
                nn.Conv1d(fnn_input_size, opt['hidden_size']*4, 1, 1, bias=False),
                nn.BatchNorm1d( opt['hidden_size']*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*4, 1, 1, bias=False),
                nn.BatchNorm1d( opt['hidden_size']*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*4, 1, 1, bias=False),
                nn.BatchNorm1d( opt['hidden_size']*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*4, 1, 1, bias=False),
                nn.BatchNorm1d( opt['hidden_size']*4),
                nn.LeakyReLU(0.2, inplace=True),
#                nn.Dropout(0.1),
                nn.Conv1d(opt['hidden_size']*4, 2, 1, 1, bias=False),
            )
        else:
            raise Exception('Architecture undefined!')
        
        if opt['focal_alpha'] != 0:
            self.fl_loss = FocalLoss1d(2, gamma=opt['focal_gamma'], alpha=opt['focal_alpha'])
        else:
            self.fl_loss = None

        # TODO: DISCUSS WITH WAYNE; for now when opt['neg_ratio'] <= 0, just use default weight
        # originally: self.ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1/opt['neg_ratio']]))
        if opt['neg_ratio'] > 0:
            self.ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1/opt['neg_ratio']]))
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.apply(self.weights_init)
        self.count=0

    def rank_cand_select(self, cands_ans_pos, scores, batch_size):
        n_cands = cands_ans_pos.size(0)
        assert n_cands % batch_size == 0, "Error: total n_cands should be multiple of batch_size"
        n_cands_per_batch = round(n_cands / batch_size)
        
        predict_s = []
        predict_e = []
        for i in range(batch_size):
            base_idx = i*n_cands_per_batch
            score, idx = scores[base_idx:base_idx+n_cands_per_batch].max(dim=0)
            _predict = cands_ans_pos[base_idx+idx.item()]
            _predict_s = round(_predict[0].item())
            _predict_e = round(_predict[1].item())
            #   Round float to int, b/c _predict_s & _predict_s is an index (so it was a whole number with float type).
            predict_s.append(_predict_s)
            predict_e.append(_predict_e)
        return predict_s, predict_e
            
    def sample_via_fofe_tricontext(self, doc_emb, query_emb, doc_mask, query_mask, target_s=None, target_e=None, test_mode=False):
        train_mode = (target_s is not None) and (target_e is not None)
        n_fofe_alphas = len(self.opt['fofe_alpha'])
        dq_fofes = []
        # 1. Construct FOFE Doc & Query Inputs Matrix
        if test_mode:
            doc_fofe, _cands_ans_pos, _padded_cands_mask = self.doc_fofe_tricontext_encoder(doc_emb, doc_mask, test_mode)
        else:
            doc_fofe = self.doc_fofe_tricontext_encoder(doc_emb, doc_mask, test_mode)
        dq_fofes.append(doc_fofe)
        batch_size = doc_fofe.size(0)
        n_cands_ans = doc_fofe.size(1)
        doc_embedding_dim = doc_fofe.size(-1) / n_fofe_alphas
        
        query_fofe = self.query_fofe_encoder(query_emb, query_mask, batch_size, n_cands_ans)
        dq_fofes.append(query_fofe)       
        query_embedding_dim = query_fofe.size(-1) / n_fofe_alphas
        
        dq_input = torch.cat(dq_fofes, dim=-1)\
                    .view([batch_size*n_cands_ans,(query_embedding_dim+doc_embedding_dim)*n_fofe_alphas])

        # 2. In train_mode: Build Target Scores Matrix, and then Sampling.
        if train_mode:
            target_score = doc_emb.new_zeros(dq_input.size(0)).unsqueeze(-1)
            _samples_idx = []
            
            # sample_num, n_neg_samples, n_pos_samples are number of samples per batch.
            if self.opt['sample_num'] > 0 and self.opt['neg_ratio'] > 0:
                sample_num = self.opt['sample_num']
                n_neg_samples = round(sample_num * self.opt['neg_ratio'])
                n_pos_samples = sample_num - n_neg_samples
            elif self.opt['sample_num'] <= 0 and self.opt['neg_ratio'] > 0:
                n_neg_samples = n_cands_ans - 1
                n_pos_samples = round(n_neg_samples / self.opt['neg_ratio']) - n_neg_samples
                sample_num = n_pos_samples + n_neg_samples
            elif self.opt['sample_num'] <= 0 and self.opt['neg_ratio'] <= 0:
                sample_num = n_cands_ans
                n_pos_samples = 1
                n_neg_samples = sample_num - n_pos_samples
            else:
                sample_num = self.opt['sample_num']
                n_pos_samples = 1
                n_neg_samples = sample_num - n_pos_samples
        
            for i in range(target_s.size(0)):
                # 2.1. Build Target Scores Matrix.
                ans_s = target_s[i].item()
                ans_e = target_e[i].item()
                ans_span = ans_e - ans_s
                doc_len = min(doc_emb.size(1), self.doc_fofe_tricontext_encoder.fofe_encoders[0].doc_len_limit)
                max_cand_len = min(doc_len, self.doc_fofe_tricontext_encoder.fofe_encoders[0].cand_len_limit)
                assert max_cand_len >= ans_span, ("max_cand_len should alway be > cand_len/ans_span; "
                                                  "CURRENT: max_cand_len = {0}, ans_span = {1}".format(max_cand_len, ans_span))
                assert doc_len >= max_cand_len, ("doc_len should alway be > max_cand_len; "
                                                  "CURRENT: doc_len = {0}, max_cand_len = {1}".format(doc_len, max_cand_len))

                currbatch_base_idx = i * n_cands_ans
                nextbatch_base_idx = (i+1) * n_cands_ans
                ans_idx = self.doc_fofe_tricontext_encoder.fofe_encoders[0].forward_fofe\
                            .get_sample_idx(ans_s, ans_span, doc_len, max_cand_len, currbatch_base_idx)
                target_score[ans_idx] = 1
                
                # 2.2. Sampling
                #   NOTED: n_pos_samples and n_neg_samples are number of pos/neg samples PER BATCH.
                #   TODO @SED: more efficient approach; current sampling method waste via python list, then convert it to equivalent tensor.
                if n_pos_samples == 1 and sample_num == n_cands_ans:
                    currbatch_samples_idx = list(range(currbatch_base_idx, nextbatch_base_idx))
                else:
                    neg_samples_population = list(range(currbatch_base_idx, ans_idx)) + list(range(ans_idx+1, nextbatch_base_idx))
                    n_neg_samples_quot, n_neg_samples_mod = divmod(n_neg_samples, len(neg_samples_population))
                    currbatch_samples_idx = ([ans_idx] * n_pos_samples) + \
                                            (random.sample(neg_samples_population, n_neg_samples_mod)) + \
                                            (neg_samples_population * n_neg_samples_quot)
                random.shuffle(currbatch_samples_idx)
                _samples_idx += currbatch_samples_idx
                
            samples_idx = dq_input.new_tensor(_samples_idx, dtype=torch.long)
            samples_dq_input = dq_input.index_select(0, samples_idx)
            samples_target_score = target_score.index_select(0, samples_idx)

        # 2. In test_mode: Build batchwise cands_ans_pos and padded_cands_mask.
        if test_mode:
            # 2.1. Reshape batchwise cands_ans_pos and padded_cands_mask (i.e. stack each batch up)
            cands_ans_pos = _cands_ans_pos.contiguous().view([batch_size*n_cands_ans,_cands_ans_pos.size(-1)])
            padded_cands_mask = _padded_cands_mask.contiguous().view([batch_size*n_cands_ans, 1])

        # 3. Determine what to return base on mode
        #    NOTE: also reshape dq_input and target_score to match conv1d (instead of linear)
        if (train_mode) and (not test_mode):
            # 3.1. Train Mode
            samples_dq_input = samples_dq_input.transpose(-1,-2).unsqueeze(0)
            samples_target_score = samples_target_score.transpose(-1,-2).long()
            return samples_dq_input, samples_target_score
        elif (not train_mode) and (test_mode):
            # 3.2. Test Mode
            dq_input = dq_input.transpose(-1,-2).unsqueeze(0)
            return dq_input, cands_ans_pos, padded_cands_mask
        elif (train_mode) and (test_mode):
            # 3.3. Draw Score Mode (aka for debuging)
            dq_input = dq_input.transpose(-1,-2).unsqueeze(0)
            target_score = target_score.transpose(-1,-2).long()
            
            #*****TODO @SED: add debugger to check target_score match with target_s & target_e.*****
            #import pdb;pdb.set_trace()
            return dq_input, target_score, cands_ans_pos, padded_cands_mask
        else:
            raise ValueError("This is supervise learning, must have target during training; invalid values:\n \
                             test_mode={0}, target_s={1}, target_e={2}".format(test_mode, target_s, target_e))

    #--------------------------------------------------------------------------------
           
    def scan_all(self, doc_emb, doc_mask, query_emb, query_mask, target_s=None, target_e=None):
        doc_emb = doc_emb.transpose(-2,-1)
        doc_len = doc_emb.size(-1)
        batchsize = doc_emb.size(0)

        if target_s is not None:
            ans_span = target_e - target_s
            v, idx = torch.max(ans_span, dim=0)
            max_len = int(min(max(self.opt['max_len'], v+1), doc_len))
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
        mask_batch = []
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
                    ans_e = target_e[j].item()+1
                    ans_s = target_s[j].item()+1
                    ans_len = ans_span[j].item()+1
                    if ans_len == i+1:
                        can_score[j, :, i, ans_e:ans_s+i+1].fill_(1)
                score_batch.append(can_score[:, :, i, 1+i:doc_len+1])

            mask_batch.append(doc_mask[:, i:])
            starts.append(torch.arange(0, doc_len-i, device=doc_emb.device))
            ends.append(torch.arange(i, doc_len, device=doc_emb.device))

        l_ctx_in_batch =torch.cat(l_ctx_in_batch, dim=-1)
        l_ctx_ex_batch =torch.cat(l_ctx_ex_batch, dim=-1)
        r_ctx_ex_batch =torch.cat(r_ctx_ex_batch, dim=-1)
        r_ctx_in_batch =torch.cat(r_ctx_in_batch, dim=-1)
        inverse_ans_batch = torch.cat(inverse_ans_batch, dim=-1)
        forward_ans_batch = torch.cat(forward_ans_batch, dim=-1)

        # generate query batch
        query_fofe = self.fofe_linear(query_emb, query_mask.float())
        query_batch = []
        for i in range(forward_ans_batch.size(-1)):
            query_batch.append(query_fofe)
        query_batch = torch.cat(query_batch, dim=-1)  
        dq_input = torch.cat([forward_ans_batch, inverse_ans_batch, l_ctx_ex_batch, l_ctx_in_batch, r_ctx_ex_batch, r_ctx_in_batch, query_batch], dim=1)

        mask_batch = torch.cat(mask_batch, dim=-1)
        starts = torch.cat(starts, dim=0).long()
        ends = torch.cat(ends, dim=0).long()
        if target_s is not None:
            target_score = torch.cat(score_batch, dim=-1).squeeze(1).long()
            return dq_input, target_score, starts, ends, mask_batch

        return dq_input, starts, ends, mask_batch
      

    def rank_select(self, scores, starts, ends):
        batchsize = scores.size(0)
        s_idxs = []
        e_idxs = []       
        for i in range(batchsize):
            v, idx = torch.max(scores[i], dim=0)
            # if v < 0.5:
            #     s_idxs.append(-1)
            #     e_idxs.append(-1)
            # else:
            s_idxs.append(starts[idx.item()].item())                                                                                                                                                    
            e_idxs.append(ends[idx.item()].item())
        return s_idxs, e_idxs                    

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
            # dq_input, target_score, starts, ends, mask_batch = self.scan_all(doc_emb, doc_mask, query_emb, query_mask, target_s, target_e)
            dq_input, target_score = self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask, target_s, target_e, test_mode=False)
            #assert target_score.sum(-1).sum(-1) == 4
            scores = self.fnn(dq_input)
            loss = self.ce_loss(scores, target_score)
            if self.fl_loss is not None:
                loss = loss + self.fl_loss(scores, target_score)
            loss = loss + F.cross_entropy(scores[:,1,:], torch.argmax(target_score, dim=1))
            # import pdb;pdb.set_trace()
            return loss
        elif self.opt['draw_score']:
            # Wayne's Version - to test target in Training vs in Data (should have 100 EM):
            # dq_input, target_score, starts, ends, d_mask = self.scan_all(doc_emb, doc_mask, query_emb, query_mask, target_s, target_e)
            # scores = self.fnn(dq_input)
            # scores = F.softmax(scores, dim=1)
            # target_score = target_score.float()
            # target_score.data.masked_fill_(d_mask.data, -float('inf'))
            # s_idxs, e_idxs = self.rank_select(target_score, starts, ends)
            
            # Sed's Version - to test target in Training vs in Data (should have 100 EM):
            # dq_input, target_score, cands_ans_pos, padded_cands_mask= self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask, target_s, target_e, test_mode=True)
            # target_score = target_score.squeeze(0).float()
            # target_score.masked_fill_(padded_cands_mask.squeeze(-1), -float('inf'))
            # batch_size = query.size(0)
            # predict_s, predict_e = self.rank_cand_select(cands_ans_pos, target_score, batch_size)
            # return predict_s, predict_e
            
            dq_input, target_score, cands_ans_pos, padded_cands_mask= self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask, target_s, target_e, test_mode=True)
            score = self.fnn(dq_input)
            score = F.softmax(score, dim=1)
            score = score[:,1,:].squeeze(0)
            target_score = target_score.squeeze(0)
            return score, target_score, cands_ans_pos, padded_cands_mask
        else:
            # Wayne's Version:
            # dq_input, starts, ends, d_mask = self.scan_all(doc_emb, doc_mask, query_emb, query_mask)
            # scores = self.fnn(dq_input)
            # scores = F.softmax(scores, dim=1)
            # score = scores[:,1,:].squeeze(0)
            # scores[:,1,:].data.masked_fill_(d_mask.data, -float('inf'))
            # s_idxs, e_idxs = self.rank_select(scores[:,1,:], starts, ends)
            # return s_idxs, e_idxs

            dq_input, cands_ans_pos, padded_cands_mask = self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask, test_mode=True)
            scores = self.fnn(dq_input)
            scores = scores[:,1,:].squeeze(0)
            scores.masked_fill_(padded_cands_mask.squeeze(-1), -float('inf'))
            batch_size = query.size(0)
            predict_s, predict_e = self.rank_cand_select(cands_ans_pos, scores, batch_size)
            return predict_s, predict_e

