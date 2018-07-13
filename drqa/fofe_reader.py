# Modification:
#  -change to support fofe_nn
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from .fofe_modules import fofe_conv1d, fofe, fofe_block, fofe_res_block, fofe_encoder, fofe_linear_tricontext
from .fofe_net import FOFENet, FOFE_NN
from .utils import tri_num
from .focal_loss import FocalLoss1d


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
        #----------------------------------------------------------------------------
        self.fofe_encoder = fofe_encoder(doc_input_size, opt['fofe_alpha'], opt['fofe_max_length'])
        # NOTED: current doc_len_limit = 809
        n_ctx_types = 1
        if (self.opt['contexts_incl_cand']):
            n_ctx_types += 2
        if (self.opt['contexts_excl_cand']):
            n_ctx_types += 2
        """self.fofe_tricontext_encoder = fofe_linear_tricontext(doc_input_size,
                                                              opt['fofe_alpha'],
                                                              cand_len_limit=self.opt['max_len'],
                                                              doc_len_limit=809,
                                                              has_lr_ctx_cand_incl=self.opt['contexts_incl_cand'],
                                                              has_lr_ctx_cand_excl=self.opt['contexts_excl_cand'])"""
        self.fofe_linear = fofe(opt['embedding_dim'], opt['fofe_alpha'])
        """
        self.fnn = nn.Sequential(
            nn.Linear(doc_input_size*3+opt['embedding_dim'], opt['hidden_size']*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(opt['hidden_size']*4, opt['hidden_size']*2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(opt['hidden_size']*2, 1, bias=False),
            nn.Sigmoid()
        )
        """
        self.fnn = nn.Sequential(
            nn.Conv1d(doc_input_size*3+opt['embedding_dim'], opt['hidden_size']*4, 1, 1, bias=False),
            nn.BatchNorm1d( opt['hidden_size']*4),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*2, 1, 1, bias=False),
            nn.BatchNorm1d( opt['hidden_size']*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*2, 2, 1, 1, bias=False),
            #nn.Sigmoid()
        )
        self.neg_ratio=torch.Tensor([0.2, 0.8]).cuda()
        self.focal_loss = FocalLoss1d(2, gamma=2, alpha=torch.Tensor([0.75, 0.25]).unsqueeze(-1))
    
    #--------------------------------------------------------------------------------

    def rank_tri_select(self, cands_ans_pos, scores, rejection_threshold=0.5):
        batch_size = self.opt['batch_size']
        n_cands = cands_ans_pos.size(0)
        assert n_cands % batch_size == 0, "Error: total n_cands should be multiple of batch_size"
        n_cands_per_batch = round(n_cands / batch_size)
        
        predict_s = []
        predict_e = []
        for i in range(batch_size):
            base_idx = i*n_cands_per_batch
            score, idx = scores[base_idx:base_idx+n_cands_per_batch].max(dim=0)
            
            # TODO @SED: ADD REJECTION MECHANISM
            #if score < rejection_threshold:
            #    _predict_s = -1
            #    _predict_e = -1
            #else:
            #    _predict = cands_ans_pos[base_idx+idx.item()]
            #    _predict_s = round(_predict[0].item())
            #    _predict_e = round(_predict[1].item())
            
            _predict = cands_ans_pos[base_idx+idx.item()]
            _predict_s = round(_predict[0].item())
            _predict_e = round(_predict[1].item())
            #   Round float to int, b/c _predict_s & _predict_s is an index (so it was a whole number with float type).
            predict_s.append(_predict_s)
            predict_e.append(_predict_e)
        return predict_s, predict_e

    def sample_via_fofe_tricontext(self, doc_emb, query_emb, target_s=None, target_e=None):
        # TODO @SED [PRIORITY 1]: FIX PADDING / BATCHSIZE>1 ISSUE in DOC_FOFE.
        # TODO @SED [PRIORITY 2]: FIX PADDING / BATCHSIZE>1 ISSUE in QUERY_FOFE.
        _doc_fofe, _cands_ans_pos = self.fofe_tricontext_encoder(doc_emb)
        _query_fofe = self.fofe_linear(query_emb)
        # import pdb;pdb.set_trace()
        batch_size = _query_fofe.size(0)
        query_embedding_dim = _query_fofe.size(-1)
        doc_embedding_dim = _doc_fofe.size(-1)
        n_cands_ans = _doc_fofe.size(1)

        # 1. Construct FOFE Doc & Query Inputs Matrix
        query_fofe = _query_fofe.new_empty(batch_size,n_cands_ans,query_embedding_dim)
        query_fofe.copy_(_query_fofe.unsqueeze(1).expand(batch_size,n_cands_ans,query_embedding_dim))
        dq_input = _doc_fofe.new_empty(batch_size*n_cands_ans,query_embedding_dim+doc_embedding_dim)
        dq_input.copy_(torch.cat([_doc_fofe, query_fofe], dim=-1).view([batch_size*n_cands_ans,query_embedding_dim+doc_embedding_dim]))

        doc_len = min(doc_emb.size(1), self.fofe_tricontext_encoder.doc_len_limit)
        max_cand_len = self.fofe_tricontext_encoder.cand_len_limit
        
        if self.training:
            assert (target_s is not None) and (target_e is not None), "This is supervise learning, must have target during training"
            # 2. & 3.
            target_score = doc_emb.new_zeros(dq_input.size(0)).unsqueeze(-1)
            _samples_idx = []
            n_neg_samples = round(self.opt['sample_num'] * self.opt['neg_ratio'])
            n_pos_samples = self.opt['sample_num'] - n_neg_samples
            for i in range(doc_emb.size(0)):
                # 2. Build Target Scores Matrix.
                ans_s = target_s[i].item()
                ans_e = target_e[i].item()
                ans_span = ans_e - ans_s
                currbatch_base_idx = i * n_cands_ans
                nextbatch_base_idx = (i+1) * n_cands_ans
                def get_sample_index(ans_s, ans_span, doc_len, max_cand_len):
                    if (ans_s < doc_len - max_cand_len):
                        ans_base_idx = ans_s * max_cand_len
                    else:
                        rev_ans_s = doc_len - ans_s - 1
                        base_idx_of_ans_base_idx = (doc_len - max_cand_len) * max_cand_len
                        ans_base_idx = base_idx_of_ans_base_idx + tri_num(max_cand_len) - tri_num(rev_ans_s+1)
                    ans_idx = currbatch_base_idx + ans_base_idx + ans_span
                    return ans_idx
                ans_idx = get_sample_index(ans_s, ans_span, doc_len, max_cand_len)
                target_score[ans_idx] = 1
                
                # 3. Sampling
                #   NOTED: n_pos_samples and n_neg_samples are number of pos/neg samples PER BATCH.
                #   TODO @SED: more efficient approach; current sampling method waste via python list, then convert it to equivalent tensor.
                neg_samples_population = list(range(currbatch_base_idx, ans_idx)) + list(range(ans_idx+1, nextbatch_base_idx))
                n_neg_samples_quot, n_neg_samples_mod = divmod(n_neg_samples, len(neg_samples_population))
                currbatch_samples_idx = ([ans_idx] * n_pos_samples) + \
                                        (random.sample(neg_samples_population, n_neg_samples_mod)) + \
                                        (neg_samples_population * n_neg_samples_quot)
                random.shuffle(currbatch_samples_idx)
                _samples_idx += currbatch_samples_idx
            samples_idx = dq_input.new_tensor(_samples_idx, dtype=torch.long)
            samples_dq_input = dq_input.new_empty(batch_size * self.opt['sample_num'], query_embedding_dim+doc_embedding_dim)
            samples_dq_input.copy_(dq_input.index_select(0, samples_idx))
            samples_target_score = target_score.new_empty(batch_size * self.opt['sample_num'], 1)
            samples_target_score.copy_(target_score.index_select(0, samples_idx))

            #import pdb;pdb.set_trace()
            return samples_dq_input, samples_target_score
        else:
            cands_ans_pos = _cands_ans_pos.new_empty(batch_size*n_cands_ans,_cands_ans_pos.size(-1))
            cands_ans_pos.copy_(_cands_ans_pos.contiguous().view([batch_size*n_cands_ans,_cands_ans_pos.size(-1)]))
            return dq_input, cands_ans_pos

    #--------------------------------------------------------------------------------
    def gen_random(self, doc_len):
        rand_length = torch.randperm(min(self.opt['max_len'], doc_len-2))
        negtive_num = self.opt['sample_num']*self.opt['neg_ratio']
        pos_num = negtive_num//rand_length.size(0)+1
        rand_position = []
        for i in range(rand_length.size(0)):
            rand_ans_length = rand_length[i].item()
            if doc_len-rand_ans_length >= pos_num:
                rand_position.append(random.sample(list(range(doc_len-rand_ans_length)), int(pos_num)))   
            else:
                rand_position.append(torch.randint(0, doc_len-rand_ans_length, (pos_num,), dtype=torch.long))    

        return rand_length, rand_position

    def sample(self, doc_emb, query_emb, target_s, target_e):
        doc_emb = doc_emb.transpose(-2,-1)
        forward_fofe, inverse_fofe = self.fofe_encoder(doc_emb)
        
        # generate positive ans and ctx batch
        ans_span = target_e - target_s
        positive_num = int(self.opt['sample_num']*(1-self.opt['neg_ratio']))
        batchsize = doc_emb.size(0)
        ans_minibatch = []
        l_ctx_minibatch = []
        r_ctx_minibatch = []
        for j in range(batchsize):
            ans_minibatch.append(forward_fofe[j, :, ans_span[j].item(), target_e[j].item()+1].unsqueeze(0))
            l_ctx_minibatch.append(forward_fofe[j, :, -1, target_s[j].item()].unsqueeze(0))
            r_ctx_minibatch.append(inverse_fofe[j, :, -1, target_e[j].item()+1].unsqueeze(0))
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
        rand_length, rand_position = self.gen_random(doc_emb.size(-1))   
        negtive_num = self.opt['sample_num']*self.opt['neg_ratio']   
        # import pdb; pdb.set_trace()
        for i in range(rand_length.size(0)):
            rand_ans_length = rand_length[i].item()
            rand_l_position = doc_emb.new_tensor(rand_position[i], dtype=torch.long)
            rand_ans_position = rand_l_position + 1 + rand_ans_length
            rand_r_position = rand_l_position + 1 + rand_ans_length
            rand_ans.append(torch.index_select(forward_fofe[:, :, rand_ans_length, :], dim=-1, index=rand_ans_position))
            rand_l_ctx.append(torch.index_select(forward_fofe[:, :, -1, :], dim=-1, index=rand_l_position))
            rand_r_ctx.append(torch.index_select(inverse_fofe[:, :, -1, :], dim=-1, index=rand_r_position))
        
        neg_ans = torch.cat(rand_ans, dim=-1)
        neg_l_ctx = torch.cat(rand_l_ctx, dim=-1)
        neg_r_ctx = torch.cat(rand_r_ctx, dim=-1)
        negtive_ctx_ans = torch.cat([neg_l_ctx, neg_ans, neg_r_ctx], dim=1)
        negtive_score = doc_emb.new_zeros((negtive_ctx_ans.size(0), 1, negtive_ctx_ans.size(-1)))
        
        # generate query batch
        query_fofe = self.fofe_linear(query_emb).unsqueeze(-1)
        query_batch = []
        for i in range(self.opt['sample_num']):
            query_batch.append(query_fofe)
        query_batch = torch.cat(query_batch, dim=-1)
        
        # generate net input and target with shuffle
        idx = doc_emb.new_tensor(torch.randperm(self.opt['sample_num']), dtype=torch.long)
        ctx_ans = torch.index_select(torch.cat([positive_ctx_ans, negtive_ctx_ans], dim=-1), dim=-1, index=idx)
        target_score =torch.index_select(torch.cat([positive_score, negtive_score], dim=-1), dim=-1, index=idx)
        #ctx_ans = torch.cat([positive_ctx_ans, negtive_ctx_ans], dim=-1)
        #target_score = torch.cat([positive_score, negtive_score], dim=-1)
        dq_input = torch.cat([ctx_ans, query_batch], dim=1)
        
        return dq_input, target_score
    
    def scan_all(self, doc_emb, query_emb, doc_mask, target_s=None, target_e=None):
        doc_emb = doc_emb.transpose(-2,-1)
        forward_fofe, inverse_fofe = self.fofe_encoder(doc_emb)
        doc_len = doc_emb.size(-1)
        batchsize = doc_emb.size(0)

        if self.training:
            ans_span = target_e - target_s
            v, idx = torch.max(ans_span, dim=0)
            max_len = int(max(self.opt['max_len'], v+1))
            score_batch = []
            can_score = doc_emb.new_zeros((batchsize, 1, max_len, doc_len+1))
        else:
            max_len = int(min(self.opt['max_len'], doc_len))

        # generate ctx and ans batch
        l_ctx_batch = []
        r_ctx_batch = []
        ans_batch = []
        mask_batch = []
        starts = []
        ends = []
        for i in range(max_len):
            l_ctx_batch.append(forward_fofe[:, :, -1, 0:doc_len-i])
            r_ctx_batch.append(inverse_fofe[:, :, -1, i+1:doc_len+1])
            ans_batch.append(forward_fofe[:, :, i, 1+i:doc_len+1])
            if self.training:
                for j in range(batchsize):
                    ans_e = target_e[j].item()+1
                    ans_s = target_s[j].item()+1
                    ans_len = ans_span[j].item()+1
                    if ans_len == i+1:
                        can_score[j, :, i, ans_e:ans_s+i+1].fill_(ans_len/(i+1))
                    #    for k in range(ans_len):
                    #        can_score[j, :, i, max(ans_e-k-1, 0)].fill_((ans_len - k - 1)/(i + k + 1))
                    #        can_score[j, :, i, min(ans_s+i+k+1, doc_len)].fill_((ans_len - k - 1)/(i + k + 1))
                    #else:
                    #    can_score[j, :, i, ans_s+i:ans_e+1].fill_((i+1)/ans_len)
                    #    for k in range(i+1):
                    #        can_score[j, :, i, max(ans_s+i-k-1, 0)] = (i - k)/(ans_len + k + 1)
                    #        can_score[j, :, i, min(ans_e+k+1, doc_len)] = (i - k)/(ans_len + k + 1)
                #import pdb; pdb.set_trace()
                score_batch.append(can_score[:, :, i, 1+i:doc_len+1])
            else:
                mask_batch.append(doc_mask[:, i:])
                starts.append(torch.arange(0, doc_len-i, device=doc_emb.device))
                ends.append(torch.arange(i, doc_len, device=doc_emb.device))

        l_ctx_batch =torch.cat(l_ctx_batch, dim=-1)
        r_ctx_batch =torch.cat(r_ctx_batch, dim=-1)
        ans_batch = torch.cat(ans_batch, dim=-1)
        ctx_ans = torch.cat([l_ctx_batch, ans_batch, r_ctx_batch], dim=1)

        # generate query batch
        query_fofe = self.fofe_linear(query_emb).unsqueeze(-1)
        query_batch = []
        for i in range(ctx_ans.size(-1)):
            query_batch.append(query_fofe)
        query_batch = torch.cat(query_batch, dim=-1)   
        dq_input = torch.cat([ctx_ans, query_batch], dim=1)
        #import pdb; pdb.set_trace()
        if self.training:
            #import pdb; pdb.set_trace()
            target_score = torch.cat(score_batch, dim=-1).squeeze(1).long()
            return dq_input, target_score
        else:
            mask_batch = torch.cat(mask_batch, dim=-1)
            starts = torch.cat(starts, dim=0).long()
            ends = torch.cat(ends, dim=0).long()
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
        if self.training :
            #--------------------------------------------------------------------------------            
            dq_input, target_score = self.scan_all(doc_emb, query_emb, doc_mask, target_s, target_e)
            #dq_input, target_score = self.sample_via_fofe_tricontext(doc_emb, query_emb, target_s, target_e)
            scores = self.fnn(dq_input)
            scores = F.log_softmax(scores, dim=1)
            fl_loss = self.focal_loss(scores, target_score)
            #ce_loss = F.nll_loss(scores, target_score, weight=self.neg_ratio)
            #loss = fl_loss + ce_loss
            return fl_loss
        else :
            # import pdb;pdb.set_trace()
            #--------------------------------------------------------------------------------
            #dq_input, cands_ans_pos  = self.sample_via_fofe_tricontext(doc_emb, query_emb)
            #score = self.fnn(dq_input)
            #predict_s, predict_e = self.rank_tri_select(cands_ans_pos, score)
            # return predict_s, predict_e
            #--------------------------------------------------------------------------------
            dq_input, starts, ends, d_mask = self.scan_all(doc_emb, query_emb, doc_mask)
            scores = self.fnn(dq_input)
            scores = F.softmax(scores, dim=1)
            scores[:,1,:].data.masked_fill_(d_mask.data, -float('inf'))
            s_idxs, e_idxs = self.rank_select(scores[:,1,:], starts, ends)
           
            return s_idxs, e_idxs 
          
            


