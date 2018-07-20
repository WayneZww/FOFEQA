# Modification:
#  -change to support fofe_nn
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from .fofe_modules import fofe_conv1d, fofe, fofe_block, fofe_res_block, fofe_encoder, fofe_tricontext
from .fofe_net import FOFENet, FOFE_NN
from .utils import tri_num


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

        #----------------------------------------------------------------------------
        # Input size to doc_fofe_encoder: word_emb + manual_features
        doc_input_size = opt['embedding_dim']+opt['num_features']
        if opt['pos']:
            doc_input_size += opt['pos_size']
        if opt['ner']:
            doc_input_size += opt['ner_size']
        
        n_ctx_types = 1
        if (self.opt['contexts_incl_cand']):
            n_ctx_types += 2
        if (self.opt['contexts_excl_cand']):
            n_ctx_types += 2

        fofe_alphas = opt['fofe_alpha'].split(',')
        self.doc_fofe_tricontext_encoder = []
        self.query_fofe_encoder = []
        for _, alpha in enumerate(fofe_alphas):
            fofe_alpha = float(alpha)
            doc_len_limit = 809
            self.doc_fofe_tricontext_encoder.append(fofe_tricontext(doc_input_size,
                                                                    fofe_alpha,
                                                                    cand_len_limit=self.opt['max_len'],
                                                                    doc_len_limit=doc_len_limit,
                                                                    has_lr_ctx_cand_incl=opt['contexts_incl_cand'],
                                                                    has_lr_ctx_cand_excl=opt['contexts_excl_cand']))
            self.query_fofe_encoder.append(fofe(opt['embedding_dim'], fofe_alpha))
        self.doc_fofe_tricontext_encoder = nn.ModuleList(self.doc_fofe_tricontext_encoder)
        self.query_fofe_encoder = nn.ModuleList(self.query_fofe_encoder)
        
        # Input size to fofe_fnn: (doc_fofe_input * n_ctx_types + query_fofe_input) * n_fofe_alphas
        fnn_input_size = (doc_input_size * n_ctx_types + opt['embedding_dim']) * len(fofe_alphas)
        """
        self.fofe_encoder = fofe_encoder(doc_input_size, opt['fofe_alpha'], opt['fofe_max_length'])

        self.fnn = nn.Sequential(
            nn.Linear(doc_input_size*n_ctx_types+opt['embedding_dim'], opt['hidden_size']*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(opt['hidden_size']*4, opt['hidden_size']*2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(opt['hidden_size']*2, 1, bias=False),
            nn.Sigmoid()
        )
        self.fnn = nn.Sequential(
            nn.Conv1d(doc_input_size*n_ctx_types+opt['embedding_dim'], opt['hidden_size']*4, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*2, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*2, 1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        """
        self.fnn = nn.Sequential(
            nn.Conv1d(fnn_input_size, opt['hidden_size']*4, 1, 1, bias=False),
            nn.BatchNorm1d( opt['hidden_size']*4),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*4, 1, 1, bias=False),
            nn.BatchNorm1d( opt['hidden_size']*4),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*4, opt['hidden_size']*4, 1, 1, bias=False),
            nn.BatchNorm1d( opt['hidden_size']*4),
            nn.ReLU(inplace=True),
            nn.Conv1d(opt['hidden_size']*4, 2, 1, 1, bias=False)
        )
        self.count=0
    
    #--------------------------------------------------------------------------------

    def rank_cand_select(self, cands_ans_pos, scores, batch_size, rejection_threshold=0.5):
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

    def sample_via_fofe_tricontext(self, doc_emb, query_emb, doc_mask, query_mask, target_s=None, target_e=None):
        n_fofe_alphas = len(self.opt['fofe_alpha'].split(','))
        batch_size = query_emb.size(0)
        query_embedding_dim = query_emb.size(-1)
        doc_embedding_dim = 0
        n_cands_ans = 0
        dq_fofes = []

        # 1. Construct FOFE Doc & Query Inputs Matrix
        for d_fofe_encoder in self.doc_fofe_tricontext_encoder:
            _doc_fofe, _cands_ans_pos, _padded_cands_mask = d_fofe_encoder(doc_emb, doc_mask)
            doc_embedding_dim = _doc_fofe.size(-1)
            n_cands_ans = _doc_fofe.size(1)
            dq_fofes.append(_doc_fofe)
        for q_fofe_encoder in self.query_fofe_encoder:
            _query_fofe = q_fofe_encoder(query_emb, query_mask)\
                            .unsqueeze(1)\
                            .expand(batch_size,n_cands_ans,query_embedding_dim)
            dq_fofes.append(_query_fofe)
        dq_input = torch.cat(dq_fofes, dim=-1)\
                    .view([batch_size*n_cands_ans,(query_embedding_dim+doc_embedding_dim)*n_fofe_alphas])

        if self.training:
            assert (target_s is not None) and (target_e is not None), "This is supervise learning, must have target during training"
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
                # 2.1.1. Build Target Scores Matrix.
                ans_s = target_s[i].item()
                ans_e = target_e[i].item()
                ans_span = ans_e - ans_s
                doc_len = min(doc_emb.size(1), self.doc_fofe_tricontext_encoder[0].doc_len_limit)
                max_cand_len = self.doc_fofe_tricontext_encoder[0].cand_len_limit
                
                currbatch_base_idx = i * n_cands_ans
                nextbatch_base_idx = (i+1) * n_cands_ans
                ans_idx = self.doc_fofe_tricontext_encoder[0].get_sample_idx(ans_s,
                                                                             ans_span,
                                                                             doc_len,
                                                                             max_cand_len,
                                                                             currbatch_base_idx)
                target_score[ans_idx] = 1
                
                # 2.1.2. Sampling
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

            # 2.1.3. Reshape samples_dq_input and samples_target_score to work on conv1d (instead of linear)
            samples_dq_input = samples_dq_input.transpose(-1,-2).unsqueeze(0)
            samples_target_score = samples_target_score.transpose(-1,-2).long()
            
            return samples_dq_input, samples_target_score
        else:
            # 2.2.1 Reshape batchwise cands_ans_pos and padded_cands_mask (i.e. stack each batch up)
            cands_ans_pos = _cands_ans_pos.contiguous().view([batch_size*n_cands_ans,_cands_ans_pos.size(-1)])
            padded_cands_mask = _padded_cands_mask.contiguous().view([batch_size*n_cands_ans, 1])
            
            # 2.2.2. Reshape dq_input to work on conv1d (instead of linear)
            dq_input = dq_input.transpose(-1,-2).unsqueeze(0)
            
            return dq_input, cands_ans_pos, padded_cands_mask

    #--------------------------------------------------------------------------------
    """
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
        negtive_num = self.opt['sample_num']-positive_num
        rand_length = torch.randperm(min(self.opt['max_len'], doc_emb.size(-1)-2))
        pos_num = negtive_num//rand_length.size(0)+1
        for i in range(rand_length.size(0)):
            rand_ans_length = rand_length[i].item()
            rand_l_position = torch.randint(0, doc_emb.size(-1)-rand_ans_length, (pos_num,), dtype=torch.long, device=doc_emb.device)
            rand_position = rand_l_position + 1 + rand_ans_length
            rand_r_position = rand_l_position + 1 + rand_ans_length
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
        #ctx_ans = torch.cat([positive_ctx_ans, negtive_ctx_ans], dim=-1)
        #target_score = torch.cat([positive_score, negtive_score], dim=-1)
        dq_input = torch.cat([ctx_ans, query_batch], dim=1)
        
        return dq_input, target_score
    
    def scan_all(self, doc_emb, query_emb, doc_mask):
        doc_emb = doc_emb.transpose(-2,-1)
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
            l_ctx_batch.append(forward_fofe[:, :, -1, 0:length-i])
            r_ctx_batch.append(inverse_fofe[:, :, -1, i+1:length+1])
            ans_batch.append(forward_fofe[:, :, i, 1+i:length+1])
            mask_batch.append(doc_mask[:, i:])
            starts.append(torch.arange(0, length-i, device=doc_emb.device))
            ends.append(torch.arange(i, length, device=doc_emb.device))
        
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
    """
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
            #dq_input, target_score = self.sample(doc_emb, query_emb, target_s, target_e)
            dq_input, target_score = self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask, target_s, target_e)
            score = self.fnn(dq_input)
            score = F.log_softmax(score, dim=1)
            loss = F.nll_loss(score, target_score)
            #import pdb;pdb.set_trace()
            return loss
        else :
            #--------------------------------------------------------------------------------
            dq_input, cands_ans_pos, padded_cands_mask  = self.sample_via_fofe_tricontext(doc_emb, query_emb, doc_mask, query_mask)
            score = self.fnn(dq_input)
            score = F.softmax(score, dim=1)
            score = score[:,1:,:].squeeze(0).squeeze(0)
            score.masked_fill_(padded_cands_mask.squeeze(-1), -float('inf'))
            batch_size = query.size(0)
            predict_s, predict_e = self.rank_cand_select(cands_ans_pos, score, batch_size)
            return predict_s, predict_e
            #--------------------------------------------------------------------------------
            
            """
            import pdb;pdb.set_trace()
            dq_input, starts, ends, d_mask = self.scan_all(doc_emb, query_emb, doc_mask)
            scores = self.fnn(dq_input).squeeze(1)
            scores.data.masked_fill_(d_mask.data, -float('inf'))
            s_idxs, e_idxs = self.rank_select(scores, starts, ends)
           
            return s_idxs, e_idxs 
            """

            


