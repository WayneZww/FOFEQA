# Modification:
#  -change to support fofe_nn
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import random
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
            

from .fofe_modules import fofe_multi, fofe_multi_encoder, fofe, sed_fofe, fofe_filter, fofe_flex_all, fofe_flex_all_filter, \
                            bidirect_fofe_tricontext, bidirect_fofe, bidirect_fofe_multi_tricontext, bidirect_fofe_multi

from .fofe_net import FOFE_NN_att, FOFE_NN
from .utils import f1_score_word_lvl
from .focal_loss import FocalLoss1d


class FOFEReader(nn.Module):
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        # good initlize linear
        #if classname.find('Conv') != -1:
        #    nn.init.kaiming_normal_(m.weight.data)
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
                nn.Linear(opt['hidden_size']*4, 3),
            )
        else:
            raise Exception('Architecture undefined!')
        
        if opt['focal_alpha'] != 0:
            self.fl_loss = FocalLoss1d(2, gamma=opt['focal_gamma'], alpha=opt['focal_alpha'])
        else:
            self.fl_loss = None

        if opt['neg_ratio'] > 0:
            self.ce_losses = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1/opt['neg_ratio']]), ignore_index=-1, reduce=False)
        else:
            self.ce_losses = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
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
    
    
    @staticmethod
    def get_nearest_sent_s_and_e(target_idx, target_batch, sent_boundary):
        """
            sent_boundary - is a tensor of sentence boundaries where
                            sent_boundary[:,1] is boundary index, and
                            sent_boundary[:,0] is boundary batch
        """
        n_sent_boundaries__all_batch = sent_boundary.size(0)
        prev_sent_boundary_batch = 0
        prev_sent_boundary_idx = -1
        for i in range(n_sent_boundaries__all_batch):
            curr_sent_boundary_batch = sent_boundary[i, 0].item()
            curr_sent_boundary_idx = sent_boundary[i, 1].item()
            if prev_sent_boundary_batch != curr_sent_boundary_batch:
                # if prev sent boundary is in the last batch, then set it to curr batch and reset prev_sent_boundary_idx to -1
                prev_sent_boundary_batch = curr_sent_boundary_batch
                prev_sent_boundary_idx = -1
            if target_batch == curr_sent_boundary_batch:
                # if curr batch is the correct one, then check for matching boundary index
                if prev_sent_boundary_idx < target_idx and curr_sent_boundary_idx >= target_idx:
                    sent_s = prev_sent_boundary_idx+1
                    sent_e = curr_sent_boundary_idx
                    return sent_s, sent_e
            prev_sent_boundary_batch = curr_sent_boundary_batch
            prev_sent_boundary_idx = curr_sent_boundary_idx
        return None, None


    def sample_via_fofe_tricontext(self, doc_emb, query_emb, doc_mask, query_mask, doc_eos,
                                   doc_for_calc_f1=None, target_s=None, target_e=None, test_mode=False):
        train_mode = (target_s is not None) and (target_e is not None)
        sent_boundary_idxs = (doc_eos==1).nonzero()
        n_fofe_alphas = len(self.opt['fofe_alpha'])
        dq_fofes = []
        # 1. Construct FOFE Doc & Query Inputs Matrix. --------------------------------------------------------------------
        if test_mode:
            doc_fofe, _cands_ans_pos, _cands_tobe_mask = self.doc_fofe_tricontext_encoder(doc_emb, doc_mask, doc_eos, test_mode)
        else:
            doc_fofe = self.doc_fofe_tricontext_encoder(doc_emb, doc_mask, doc_eos, test_mode)
        dq_fofes.append(doc_fofe)
        batch_size = doc_fofe.size(0)
        n_cands_ans = doc_fofe.size(1)
        doc_embedding_dim = doc_fofe.size(-1) / n_fofe_alphas
        
        query_fofe = self.query_fofe_encoder(query_emb, query_mask, batch_size, n_cands_ans)
        dq_fofes.append(query_fofe)       
        query_embedding_dim = query_fofe.size(-1) / n_fofe_alphas
        
        dq_input = torch.cat(dq_fofes, dim=-1)\
                    .view([batch_size*n_cands_ans,(query_embedding_dim+doc_embedding_dim)*n_fofe_alphas])

        # 2. In train_mode: Build Target Scores Matrix, and then Sampling. ------------------------------------------------
        if train_mode:
            target_scores = doc_emb.new_zeros(dq_input.size(0)).unsqueeze(-1)
            f1_scores = doc_emb.new_zeros(dq_input.size(0))
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
                # 2.1.1. get parameter value required to find ans_idx
                ans_s = target_s[i].item()
                ans_e = target_e[i].item()
                ans_span = ans_e - ans_s    # NOTED: ans with 1 word will have ans_span = 0 (not 1).
                doc_len = min(doc_emb.size(1), self.doc_fofe_tricontext_encoder.fofe_encoders[0].doc_len_limit)
                max_cand_len = min(doc_len, self.doc_fofe_tricontext_encoder.fofe_encoders[0].cand_len_limit)
                if max_cand_len <= ans_span:
                    ans_span = max_cand_len - 1
                    ans_e = ans_s + ans_span
                assert max_cand_len >= ans_span + 1, ("max_cand_len should alway be > cand_len/ans_len; noted: ans_len = ans_span + 1 "
                                                      "CURRENT: max_cand_len = {0},"
                                                      "ans_span = {1}".format(max_cand_len, ans_span))
                assert doc_len >= max_cand_len, ("doc_len should alway be > max_cand_len; "
                                                 "CURRENT: doc_len = {0}, "
                                                 "max_cand_len = {1}".format(doc_len, max_cand_len))

                # 2.1.2. get parameter value required to find candidates overlapping with ans
                sent_s, sent_e = FOFEReader.get_nearest_sent_s_and_e(ans_s, i, sent_boundary_idxs)
                """
                _sent_s, _sent_e = FOFEReader.get_nearest_sent_s_and_e(ans_e, i, sent_boundary_idxs)
                assert sent_s == _sent_s and sent_e == _sent_e, ("ans_s (idx={0}) and ans_e (idx={1}) "
                                                                 "should always be in the same sentence; "
                                                                 "i.e. no cross-sentence target ans".format(ans_s, ans_e))
                assert sent_s != None and sent_e != None, ("ans_s (idx={0}) should always be "
                                                           "some where in the doc".format(ans_s))
                assert _sent_s != None and _sent_e != None, ("ans_e (idx={0}) should always be "
                                                             "some where in the doc".format(ans_e))
                """
                #list of cands who is a substring of ans
                sub_ans = [(s,e,e-s) for s in range(ans_s, ans_e+1)
                           for e in range(s, ans_e+1)
                           if (s != ans_s or e != ans_e)]
                #list of cands who is a superstring of ans
                super_ans = [(s,e,e-s) for s in range(sent_s, ans_s+1)
                             for e in range(ans_e, sent_e+1)
                             if (s != ans_s or e != ans_e)]
                #list of cands who has an intersected string with (front substring of) ans
                front_relative_ans = [(s,e,e-s) for s in range(sent_s, ans_s)
                                      for e in range(ans_s, ans_e)
                                      if (s != ans_s or e != ans_e)]
                #list of cands who has an intersected string with (rear substring of) ans
                rear_relative_ans = [(s,e,e-s) for s in range(ans_s+1, ans_e+1)
                                    for e in range(ans_e+1, sent_e+1)
                                    if (s != ans_s or e != ans_e)]
                overlapping_ans = sub_ans + super_ans + front_relative_ans + rear_relative_ans
                overlapping_ans = [(s,e,span) for s,e,span in overlapping_ans if span <= max_cand_len-1]
                
                # 2.1.3. get ans_idx, then set target_scores
                def func_get_sample_idx(sample_start_idx, sample_span, doc_len, max_cand_len, currbatch_base_idx):
                    return self.doc_fofe_tricontext_encoder\
                            .fofe_encoders[0]\
                            .forward_fofe\
                            .get_sample_idx(sample_start_idx, sample_span, doc_len, max_cand_len, currbatch_base_idx)
                currbatch_base_idx = i * n_cands_ans
                ans_idx = func_get_sample_idx(ans_s, ans_span, doc_len, max_cand_len, currbatch_base_idx)
                target_scores[ans_idx] = 2
                
                # 2.1.4. get overlapping_ans_idx, then set target_scores and f1_scores values
                for ovlp_ans_s, ovlp_ans_e, ovlp_ans_span in overlapping_ans:
                    f1 = f1_score_word_lvl(doc_for_calc_f1[i, ovlp_ans_s:ovlp_ans_e+1],
                                           doc_for_calc_f1[i, ans_s:ans_e+1])
                    ovlp_ans_idx = func_get_sample_idx(ovlp_ans_s, ovlp_ans_span, doc_len, max_cand_len, currbatch_base_idx)
                    target_scores[ovlp_ans_idx] = 1
                    f1_scores[ovlp_ans_idx] = f1
                    assert ovlp_ans_idx != ans_idx, ("ans should not be in list of candidates overlapping with ans")
                #import pdb; pdb.set_trace()

                # 2.2. Sampling
                #   NOTED: n_pos_samples and n_neg_samples are number of pos/neg samples PER BATCH.
                #   TODO @SED: more efficient approach;
                #              current sampling method was via python list, then convert it to equivalent tensor.
                nextbatch_base_idx = (i+1) * n_cands_ans
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
            samples_target_scores = target_scores.index_select(0, samples_idx)
            samples_f1_scores = f1_scores.index_select(0, samples_idx)

        # 2. In test_mode: Build batchwise cands_ans_pos and cands_tobe_mask. -------------------------------------------
        if test_mode:
            # 2.1. Reshape batchwise cands_ans_pos and cands_tobe_mask (i.e. stack each batch up)
            cands_ans_pos = _cands_ans_pos.contiguous().view([batch_size*n_cands_ans,_cands_ans_pos.size(-1)])
            cands_tobe_mask = _cands_tobe_mask.contiguous().view([batch_size*n_cands_ans, 1])

        # 3. Determine what to return base on mode. -----------------------------------------------------------------------
        #    NOTE: also reshape dq_input and target_scores to match conv1d (instead of linear)
        if (train_mode) and (not test_mode):
            # 3.1. Train Mode
            return samples_dq_input, samples_target_scores.long().squeeze(-1), samples_f1_scores.squeeze(-1)
        elif (not train_mode) and (test_mode):
            # 3.2. Test Mode and Draw Score Mode for Dev Set
            return dq_input, cands_ans_pos, cands_tobe_mask
        elif (train_mode) and (test_mode):
            # 3.3. Draw Score Mode for Train Set (aka for debuging target_s and target_e)
            return dq_input, target_scores.long().squeeze(-1), cands_ans_pos, cands_tobe_mask
        else:
            raise ValueError("This is supervise learning, must have target during training; invalid values:\n \
                             test_mode={0}, target_s={1}, target_e={2}".format(test_mode, target_s, target_e))


    def sample_via_conv(self, doc_emb, doc_mask, query_emb, query_mask, doc_pos, target_s=None, target_e=None):
        doc_emb = doc_emb.transpose(-2,-1)
        doc_len = doc_emb.size(-1)
        batchsize = doc_emb.size(0)
        pos_tagger = doc_pos[:,:,7]

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
                    ans_e = target_e[j].item()+1
                    ans_s = target_s[j].item()+1
                    ans_len = ans_span[j].item()+1
                    if ans_len == i+1:
                        can_score[j, :, i, ans_e:ans_s+i+1].fill_(1)
                score_batch.append(can_score[:, :, i, 1+i:doc_len+1])
            # add pos tagger
            _pos_tagger = doc_emb.new_zeros(pos_tagger.shape, dtype=torch.long)
            for j in range(batchsize):
                for k in range(doc_len):
                    if pos_tagger[j, k].item() == 1 :
                        for t in range(i+1):
                            if target_e is not None and target_e[j].item() == k + t :
                                continue
                            else:
                                if k+t <= doc_len-1 :
                                     _pos_tagger[j, k+t].fill_(1)

            pos_batch.append(_pos_tagger[:, i:])
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
        pos_batch = torch.cat(pos_batch, dim=-1)
        starts = torch.cat(starts, dim=0).long().unsqueeze(-1)
        ends = torch.cat(ends, dim=0).long().unsqueeze(-1)
        cands_ans_pos = torch.cat([starts, ends], dim=-1)

        # change size
        dq_input = torch.reshape(dq_input.transpose(-1,-2), (dq_input.size(0)*dq_input.size(-1), dq_input.size(1)))
        mask_batch = torch.reshape(mask_batch, (dq_input.size(0), 1))
        pos_batch = torch.reshape(pos_batch, (dq_input.size(0), 1))
        mask = mask_batch.long() + pos_batch
        mask = torch.gt(mask, 0)
        if target_s is not None:
            target_score = torch.cat(score_batch, dim=-1).squeeze(1).long()
            #change size
            target_score = torch.reshape(target_score, (dq_input.size(0),))
            target_score = target_score - mask.long().squeeze(1)
            if torch.eq(target_score, 1).sum() != batchsize :
                import pdb; pdb.set_trace()
            return dq_input, target_score, cands_ans_pos, mask

        return dq_input, cands_ans_pos, mask
    

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


    def forward(self, doc, doc_f, doc_pos, doc_ner, doc_eos, doc_mask, query, query_mask, target_s=None, target_e=None):
        """Inputs:
        doc = document/context word indices                 [batch * len_d]
        doc_mask = document/context padding mask            [batch * len_d]
        doc_eos = document/context end of sentence tag      [batch * len_d]
        doc_f = document/context word features indices      [batch * len_d * nfeat]
            -> nfeat = 4; the 4 features are:
                1. match_origin = doc_word in query; (text's format = original).
                2. match_lower = doc_word in query; (text's format = original lowercase).
                3. match_lemma = doc_word in query; (text's format = base form of the word [see spaCy's lemma]).
                4. context_tf = doc_word's occurance / doc's len; (text's format = base form of the word).
        doc_pos = document/context POS tags                 [batch * len_d * n_pos_types]
            -> n_pos_types = 51
            -> using spaCy's Detailed Part-Of-Speech tagger.
        doc_ner = document/context entity tags              [batch * len_d * n_ent_types]
            -> n_ent_types = 19
        query = query/question word indices                 [batch * len_q]
        query_mask = query/question padding mask            [batch * len_q]
        """
        doc_emb, query_emb = self.input_embedding(doc, doc_f, doc_pos, doc_ner, query)
        if self.opt['draw_score']:
            if self.opt['version'] == 's' :
                dq_input, cands_ans_pos, cands_tobe_mask = self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask, doc_eos, test_mode=True)
            else:
                dq_input, cands_ans_pos, cands_tobe_mask = self.sample_via_conv(doc_emb, doc_mask, query_emb, query_mask, doc_pos, target_s, target_e)
            scores = self.fnn(dq_input)
            scores = F.softmax(scores, dim=1)
            scores, _ = scores[:,-2:].max(dim=1, keepdim=True)      # Get max(score_of_class1, score_of_class2) for each candidate.
            scores.masked_fill_(cands_tobe_mask, -float('inf'))     # If cands is marked to be mask, set score to -inf
            scores = scores.squeeze(-1)
            #import pdb; pdb.set_trace()
            return scores, cands_ans_pos

        if not self.training:
            if self.opt['version'] == 's' :
                dq_input, cands_ans_pos, cands_tobe_mask = self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask, doc_eos, test_mode=True)
            else:
                dq_input, cands_ans_pos, cands_tobe_mask = self.sample_via_conv(doc_emb, doc_mask, query_emb, query_mask, doc_pos)
            scores = self.fnn(dq_input)
            scores = F.softmax(scores, dim=1)
            scores, _ = scores[:,-2:].max(dim=1, keepdim=True)      # Get max(score_of_class1, score_of_class2) for each candidate.
            scores.masked_fill_(cands_tobe_mask, -float('inf'))     # If cands is marked to be mask, set score to -inf
            batch_size = query.size(0)
            predict_s, predict_e = self.rank_cand_select(cands_ans_pos, scores, batch_size)
            #import pdb; pdb.set_trace()
            return predict_s, predict_e

        if self.training:
            if self.opt['version'] == 's' :
                # s version: tricontext_fofe; cand overlapping ans treat as weighted correct (weight < 1)
                dq_input, target_scores, f1_scores = self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask, doc_eos,
                                                                                     doc, target_s, target_e, test_mode=False)
                scores = self.fnn(dq_input)
                losses = self.ce_losses(scores, target_scores)
                
                def overlap_rate_loss(ce_losses, target_scores, overlap_rates, lambda1=10, lambda2=100):
                    #loss calculation:
                    #   loss = ce_loss                              if target=0; Reject
                    #   loss = ce_loss * overlap_rate * lambda1     if target=1; Partial Accept
                    #   loss = ce_loss * lambda2                    if target=2; Accept
                    target_class0_idx = (target_scores==0).nonzero().squeeze(-1)    # CASE: candidate is totally wrong
                    target_class1_idx = (target_scores==1).nonzero().squeeze(-1)    # CASE: candidate is overlapping right ans
                    target_class2_idx = (target_scores==2).nonzero().squeeze(-1)    # CASE: candidate is totally right
                    olrate_class1_idx = (overlap_rates>0).nonzero().squeeze(-1)     # olrate_class1 = F1 score word-level for class target = 1
                    losses_class0 = ce_losses.index_select(0, target_class0_idx)
                    losses_class1 = ce_losses.index_select(0, target_class1_idx)
                    losses_class2 = ce_losses.index_select(0, target_class2_idx)
                    olrate_class1 = overlap_rates.index_select(0, olrate_class1_idx)
                    return torch.cat((losses_class0,
                                      losses_class1 * olrate_class1 * lambda1,
                                      losses_class2 * lambda2),
                                     dim=0).mean(dim=0)
                
                # IF lambda1 == 0 and lambda2 == 0, use ce_loss, ELSE use overlap_rate_loss.
                if (self.opt['olr_loss_lambda1'] == 0 and self.opt['olr_loss_lambda2'] == 0):
                    loss = losses.mean(dim=0)
                else:
                    loss = overlap_rate_loss(losses,
                                             target_scores,
                                             f1_scores,
                                             lambda1=self.opt['olr_loss_lambda1'],
                                             lambda2=self.opt['olr_loss_lambda2'])
            else:
                # w version: conv_fofe; cand overlapping ans treat as wrong
                dq_input, target_scores, cands_ans_pos, mask_batch = self.sample_via_conv(doc_emb, doc_mask, query_emb, query_mask, doc_pos, target_s, target_e)
                scores = self.fnn(dq_input)
                losses = self.ce_losses(scores, target_scores)
                loss = losses.mean(dim=0)
            if self.fl_loss is not None:
                loss = loss + self.fl_loss(scores, target_scores)
            #loss = loss + F.cross_entropy(scores[:,1,:], torch.argmax(target_scores, dim=1))
            #import pdb; pdb.set_trace()
            return loss

            


