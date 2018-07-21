import math
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .utils import tri_num, count_num_substring

class fofe_tricontext(nn.Module):
    def __init__(self, embedding_dim,  alpha, cand_len_limit=10, doc_len_limit=46, has_lr_ctx_cand_incl=True, has_lr_ctx_cand_excl=True):
        super(fofe_tricontext, self).__init__()
        self.alpha = alpha
        self.cand_len_limit = cand_len_limit
        self.has_lr_ctx_cand_incl = has_lr_ctx_cand_incl
        self.has_lr_ctx_cand_excl = has_lr_ctx_cand_excl
        
        #Construct Base FOFE Buffer for full doc length
        self.doc_len_limit = doc_len_limit
        self._full_base_block_alpha = torch.zeros(self.doc_len_limit, self.doc_len_limit)
        for i in range(1, self.doc_len_limit+1):
            powers = torch.linspace(i-1,i-self.doc_len_limit,self.doc_len_limit).abs()
            self._full_base_block_alpha[i-1,:].copy_(torch.pow(self.alpha,powers))

    @staticmethod
    def get_sample_idx(sample_start_idx, sample_span, doc_len, max_cand_len, currbatch_base_idx=0):
        '''
            sample_start_idx = starting index of target sample within the doc.
            sample_span = length of target sample.
            doc_len = length of doc that target sample was in.
            max_cand_len = a predefine max candidate length that fofe_tricontext was build for.
        '''
        
        if (sample_start_idx < doc_len - max_cand_len):
            sample_base_idx = sample_start_idx * max_cand_len
        else:
            rev_sample_start_idx = doc_len - sample_start_idx - 1
            base_idx_of_sample_base_idx = (doc_len - max_cand_len) * max_cand_len
            sample_base_idx = base_idx_of_sample_base_idx + tri_num(max_cand_len) - tri_num(rev_sample_start_idx+1)
        sample_idx = currbatch_base_idx + sample_base_idx + sample_span
        return sample_idx

    def get_contexts_alpha_buffers(self, x_input):
        '''
            Derive context_alpha_buffers from _base_tril_alpha and _base_triu_alpha
            
            context_alpha_buffers[0] = Candidate Context alpha buffer
            context_alpha_buffers[1] = Left Context (Candidate excluded) alpha buffer
            context_alpha_buffers[2] = Left Context (Candidate Included) alpha buffer
            context_alpha_buffers[3] = Right Context (Candidate excluded)alpha buffer
            context_alpha_buffers[4] = Right Context (Candidate Included) alpha buffer
        '''
        #Construct Base FOFE Buffer for specific doc length;
        doc_len = min(x_input.size(1), self.doc_len_limit)
        _base_tril_alpha = x_input.new_zeros(doc_len,doc_len)
        _base_triu_alpha = x_input.new_zeros(doc_len,doc_len)
        _base_tril_alpha.copy_(self._full_base_block_alpha.tril()[:doc_len,:doc_len])
        _base_triu_alpha.copy_(self._full_base_block_alpha.triu()[:doc_len,:doc_len])
        
        n_cand, max_cand_len = count_num_substring(self.cand_len_limit, doc_len)
        context_alpha_buffers = []
        cands_pos = x_input.new_zeros(n_cand,2)
        
        for i in range(5):
            context_alpha_buffers.append(x_input.new_zeros(n_cand,doc_len))
        
        for i in range(doc_len):
            if (i < doc_len - max_cand_len):
                start_idx = i * max_cand_len
                end_idx = (i+1) * max_cand_len
                
                # Candidate Context alphas buffer
                context_alpha_buffers[0][start_idx:end_idx,i:max_cand_len+i].copy_(_base_tril_alpha[i:max_cand_len+i,i:max_cand_len+i])
                
                # Left Context (Candidate included) alpha buffer
                context_alpha_buffers[2][start_idx:end_idx,:max_cand_len+i].copy_(_base_tril_alpha[i:max_cand_len+i,:max_cand_len+i])
                
                # Right Context (Candidate excluded) alpha buffer
                context_alpha_buffers[3][start_idx:end_idx,1:].copy_(_base_triu_alpha[i:max_cand_len+i,:-1])
        
            else:
                rev_i = doc_len-i-1
                base_idx = (doc_len - max_cand_len) * max_cand_len
                start_idx = base_idx + tri_num(max_cand_len) - tri_num(rev_i+1)
                end_idx = base_idx + tri_num(max_cand_len) - tri_num(rev_i)
                
                # Candidate Context alphas buffer
                context_alpha_buffers[0][start_idx:end_idx,i:doc_len].copy_(_base_tril_alpha[i:doc_len,i:doc_len])
                
                # Left Context (Candidate included) alpha buffer
                context_alpha_buffers[2][start_idx:end_idx,:].copy_(_base_tril_alpha[i:doc_len,:doc_len])
                
                # Right Context (Candidate excluded) alpha buffer
                context_alpha_buffers[3][start_idx:end_idx,1:].copy_(_base_triu_alpha[i:doc_len,:-1])
            
            if (i > 0):
                # SINCE: left context doesn't exist for BOS (begin of sentence) candidates,
                #   SO: leave the values as zero (when i == 0).
                context_alpha_buffers[1][start_idx:end_idx,:].copy_(_base_tril_alpha[i-1,:doc_len].expand(end_idx-start_idx,doc_len))

            # Right Context (Candidate included) alpha buffer
            context_alpha_buffers[4][start_idx:end_idx,:].copy_(_base_triu_alpha[i,:doc_len].expand(end_idx-start_idx,doc_len))

            # Candidate Positions within Doc
            cands_pos[start_idx:end_idx, 0] = i
            cands_pos[start_idx:end_idx, 1].copy_(torch.range(i, i+end_idx-start_idx-1))
        
        return context_alpha_buffers, cands_pos
    
    def forward(self, x_input, x_mask):
        batch_size = x_input.size(0)
        length = min(x_input.size(1), self.doc_len_limit)
        embedding_dim = x_input.size(2)
        
        context_alpha_buffer, cands_pos = self.get_contexts_alpha_buffers(x_input)
        n_cand = context_alpha_buffer[0].size(0)
        n_context_types = len(context_alpha_buffer)
        #context_alpha_buffers[0] = Candidate Context alpha buffer
        #context_alpha_buffers[1] = Left Context (Candidate excluded) alpha buffer
        #context_alpha_buffers[2] = Left Context (Candidate Included) alpha buffer
        #context_alpha_buffers[3] = Right Context (Candidate excluded)alpha buffer
        #context_alpha_buffers[4] = Right Context (Candidate Included) alpha buffer

        _batchwise_alpha_buffer = []
        _batchwise_fofe_codes = []
        # TODO @SED: No need to multiply all 5 contexts types; move this to IF statement below.
        for i in range(n_context_types):
            _batchwise_alpha_buffer.append(context_alpha_buffer[i].expand(batch_size, n_cand, length))
            _batchwise_fofe_codes.append(torch.bmm(_batchwise_alpha_buffer[i],x_input))

        if ( not self.has_lr_ctx_cand_incl ) and ( not self.has_lr_ctx_cand_excl ):
            batchwise_fofe_codes = _batchwise_fofe_codes[0]
        elif ( not self.has_lr_ctx_cand_incl ) and ( self.has_lr_ctx_cand_excl ):
            batchwise_fofe_codes = torch.cat([_batchwise_fofe_codes[1],
                                              _batchwise_fofe_codes[0],
                                              _batchwise_fofe_codes[3]], dim=-1)
        elif ( self.has_lr_ctx_cand_incl ) and ( not self.has_lr_ctx_cand_excl ):
            batchwise_fofe_codes = torch.cat([_batchwise_fofe_codes[2],
                                              _batchwise_fofe_codes[1],
                                              _batchwise_fofe_codes[4]], dim=-1)
        else:
            batchwise_fofe_codes = torch.cat([_batchwise_fofe_codes[1],
                                              _batchwise_fofe_codes[2],
                                              _batchwise_fofe_codes[0],
                                              _batchwise_fofe_codes[4],
                                              _batchwise_fofe_codes[3]], dim=-1)
        batchwise_cands_pos = cands_pos.unsqueeze(0).expand(batch_size,n_cand, cands_pos.size(-1))
        batchwise_padded_cands = torch.bmm(_batchwise_alpha_buffer[0], x_mask.float().unsqueeze(-1)) > 0
    
        return batchwise_fofe_codes, batchwise_cands_pos, batchwise_padded_cands

class fofe(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe, self).__init__()
        self.alpha = alpha
        
    def forward(self, x, x_mask):
        length = x.size(-2)
        
        # Construct Alphas Buffer (i.e. Matrix)
        matrix = x.new_empty(x.size(0),1,length)
        matrix[:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        
        # Adjust Alphas Buffer to accommondate the padding
        # NOTED: alpha for the padding will become > 1, but noted that their corresponding x values are 0.
        matrix_padding_rm = torch.pow(self.alpha,-1*torch.sum(x_mask,dim=1).float()).unsqueeze(-1).unsqueeze(-1)
        matrix.copy_(torch.bmm(matrix_padding_rm, matrix))
    
        fofe_code = torch.bmm(matrix,x).squeeze(-2)
        return fofe_code


