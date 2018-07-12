import math
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .utils import tri_num, count_num_substring

class fofe_conv1d(nn.Module):
    def __init__(self, emb_dims, alpha=0.9, length=1, dilation=1, inverse=False):
        super(fofe_conv1d, self).__init__()
        self.alpha = alpha
        self.length = length
        self.channels = emb_dims
        self.fofe_filter = Parameter(torch.Tensor(emb_dims,1,length))
        self.fofe_filter.requires_grad_(False)
        self._init_filter(emb_dims, alpha, length, inverse)
        self.padding = (length - 1)//2
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(self.channels,self.channels,3,1,padding=length,
                        dilation=dilation, groups=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)           
        )

    def _init_filter(self, channels, alpha, length, inverse):
        if not inverse :
            self.fofe_filter[:,:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        else :
            self.fofe_filter[:,:,].copy_(torch.pow(self.alpha,torch.range(0,length-1)))

    def forward(self, x): 
        x = torch.transpose(x,-2,-1)
        if (self.length % 2 == 0) :
            x = F.pad(x, (0,1), mode='constant', value=0)
        x = F.conv1d(x, self.fofe_filter, bias=None, stride=1, 
                        padding=self.padding, groups=self.channels)
        x = self.dilated_conv(x)
        return x

    
class fofe_filter(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = alpha
        self.fofe_kernel = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_kernel.requires_grad_(False)
        self._init_kernel(alpha, length, inverse)
        #self.padding = (length - 1)//2
        self.inverse = inverse

    def _init_kernel(self, alpha, length, inverse):
        if not inverse :
            self.fofe_kernel[:,:,].copy_(torch.pow(alpha, torch.linspace(length-1, 0, length)))
        else :
            self.fofe_kernel[:,:,].copy_(torch.pow(alpha, torch.range(0, length-1)))
    
    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0 :
            return x
        if self.inverse:
            x = F.pad(x,(0, self.length))
        else :
            x = F.pad(x,(self.length, 0))
        x = F.conv1d(x, self.fofe_kernel, bias=None, stride=1, 
                        padding=0, groups=self.channels)

        return x


class fofe_dual_filter(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_dual_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = alpha
        self.fofe_kernel_s = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_kernel_s.requires_grad_(False)
        self.fofe_kernel_l = Parameter(torch.Tensor(inplanes,1,length))
        self.fofe_kernel_l.requires_grad_(False)
        self._init_kernel(alpha, length, inverse)
        #self.padding = (length - 1)//2
        self.inverse = inverse

    def _init_kernel(self, alpha, length, inverse):
        if not inverse :
            self.fofe_kernel_s[:,:,].copy_(torch.pow(alpha, torch.linspace(length-1, 0, length)))
            self.fofe_kernel_l[:,:,].copy_(torch.pow(alpha-0.4, torch.linspace(length-1, 0, length)))
        else :
            self.fofe_kernel_s[:,:,].copy_(torch.pow(alpha, torch.range(0, length-1)))
            self.fofe_kernel_l[:,:,].copy_(torch.pow(alpha-0.4, torch.range(0, length-1)))
    
    def forward(self, x):
        if self.alpha == 1 or self.alpha == 0 :
            return x
        if self.inverse:
            x = F.pad(x,(0, self.length))
        else :
            x = F.pad(x,(self.length, 0))
        short_fofe = F.conv1d(x, self.fofe_kernel_s, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        long_fofe = F.conv1d(x, self.fofe_kernel_l, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        fofe_code = torch.cat([short_fofe, long_fofe], dim=1)
        return fofe_code


class fofe_encoder(nn.Module):
    def __init__(self, emb_dim, fofe_alpha_l, fofe_alpha_h, fofe_max_length):
        super(fofe_encoder, self).__init__()
        self.forward_filter = []
        self.inverse_filter = []
        for i in range(fofe_max_length):
            self.forward_filter.append(fofe_flex_dual_filter(emb_dim, fofe_alpha_l, fofe_alpha_h, i+1))
            self.inverse_filter.append(fofe_flex_dual_filter(emb_dim, fofe_alpha_l, fofe_alpha_h, i+1, inverse=True))

        self.forward_filter = nn.ModuleList(self.forward_filter)
        self.inverse_filter = nn.ModuleList(self.inverse_filter)

    def forward(self, x):
        forward_fofe = []
        inverse_fofe = []
        for forward_filter in self.forward_filter:
            forward_fofe.append(forward_filter(x).unsqueeze(-2))
        for inverse_filter in self.inverse_filter:
            inverse_fofe.append(inverse_filter(x).unsqueeze(-2))

        forward_fofe = torch.cat(forward_fofe, dim=-2)
        inverse_fofe = torch.cat(inverse_fofe, dim=-2)

        return forward_fofe, inverse_fofe


class fofe_block(nn.Module):
    def __init__(self, inplanes, planes, fofe_alpha=0.8, fofe_length=3, dilation=3, fofe_inverse=False):
        super(fofe_block, self).__init__()
        self,fofe_filter = fofe_filter(inplanes, fofe_alpha, fofe_length, fofe_inverse)
        self.conv = nn.Sequential(nn.Conv1d(inplanes, planes,3,1,padding=length,
                        dilation=dilation, groups=1, bias=False),
                    nn.BatchNorm1d(planes),
                    nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x): 
        x = self.fofe_filter(x)
        x = self.conv(x)
    
        return x
    

class fofe_res_block(nn.Module):
    def __init__(self, inplanes, planes, convs=3, fofe_alpha=0.9, fofe_length=3, fofe_dilation=3, downsample=None, fofe_inverse=False):
        super(fofe_res_block, self).__init__()
        self.fofe_filter = fofe_filter(inplanes, fofe_alpha, fofe_length, fofe_inverse)
        
        self.conv = []
        self.conv.append(nn.Sequential(
                            nn.Conv1d(inplanes, planes,3,1,padding=fofe_length,
                                dilation=fofe_length, groups=1, bias=False),
                            nn.BatchNorm1d(planes)))

        for i in range(1, convs):
            self.conv.append(nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                nn.Conv1d(planes, planes, 3, 1, 1, 1, bias=False),
                                nn.BatchNorm1d(planes)))

        self.conv = nn.Sequential(*self.conv)
        self.relu = nn.LeakyReLU(0.1, inplace=True) 
        self.downsample = downsample

    def forward(self, x): 
        x = self.fofe_filter(x)
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out


#--------------------------------------------------------------------------------
class fofe_linear_tricontext(nn.Module):
    def __init__(self, embedding_dim,  alpha, cand_len_limit=10, doc_len_limit=46, has_lr_ctx_cand_incl=True, has_lr_ctx_cand_excl=True):
        super(fofe_linear_tricontext, self).__init__()
        self.alpha = alpha
        self.cand_len_limit = cand_len_limit
        
        #Construct Base FOFE Buffer for full doc length
        self.doc_len_limit = doc_len_limit
        self._full_base_block_alpha = torch.zeros(self.doc_len_limit, self.doc_len_limit)
        for i in range(1, self.doc_len_limit+1):
            powers = torch.linspace(i-1,i-self.doc_len_limit,self.doc_len_limit).abs()
            self._full_base_block_alpha[i-1,:].copy_(torch.pow(self.alpha,powers))
    
        #Construct Layer base on input arg
        self.has_lr_ctx_cand_incl = has_lr_ctx_cand_incl
        self.has_lr_ctx_cand_excl = has_lr_ctx_cand_excl
        if ( not self.has_lr_ctx_cand_incl ) and ( not self.has_lr_ctx_cand_excl ):
            self.linear = nn.Sequential(nn.Linear(embedding_dim,embedding_dim, bias=False),
                                        nn.ReLU(inplace=True))
        elif has_lr_ctx_cand_incl and has_lr_ctx_cand_excl:
            self.linear = nn.Sequential(nn.Linear(embedding_dim*5,embedding_dim*5, bias=False),
                                        nn.ReLU(inplace=True))
        else:
            self.linear = nn.Sequential(nn.Linear(embedding_dim*3,embedding_dim*3, bias=False),
                                        nn.ReLU(inplace=True))
    
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
    
    def forward(self, x_input):
        # TODO @SED [PRIORITY 1]: FIX PADDING / BATCHSIZE>1 ISSUE in DOC_FOFE
        #                   IDEA: bmm(context_alpha_buffer, doc_mask) then sum each row;
        #                         this should tell the row of the cands that included padding
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
            batchwise_fofe_codes = torch.cat([_batchwise_fofe_codes[0],
                                              _batchwise_fofe_codes[1],
                                              _batchwise_fofe_codes[3]], dim=-1)
        elif ( self.has_lr_ctx_cand_incl ) and ( not self.has_lr_ctx_cand_excl ):
            batchwise_fofe_codes = torch.cat([_batchwise_fofe_codes[0],
                                              _batchwise_fofe_codes[2],
                                              _batchwise_fofe_codes[4]], dim=-1)
        else:
            batchwise_fofe_codes = torch.cat([_batchwise_fofe_codes[0],
                                              _batchwise_fofe_codes[1],
                                              _batchwise_fofe_codes[3],
                                              _batchwise_fofe_codes[2],
                                              _batchwise_fofe_codes[4]], dim=-1)
        batchwise_cands_pos = cands_pos.unsqueeze(0).expand(batch_size,n_cand, cands_pos.size(-1))
        #output = self.linear(batchwise_fofe_codes)
        return batchwise_fofe_codes, batchwise_cands_pos

#--------------------------------------------------------------------------------


class fofe_linear(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe_linear, self).__init__()
        self.alpha = alpha
        self.linear = nn.Sequential(
            nn.Linear(channels,channels, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        length = x.size(-2)
        #Should use new_empty here
        matrix = x.new_empty(x.size(0),1,length)
        #if x.data.is_cuda :
        #    matrix = matrix.cuda()
        matrix[:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        fofe_code = torch.bmm(matrix,x)
        output = self.linear(fofe_code)
        return output


class fofe(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        length = x.size(-2)
        matrix = x.new_empty(x.size(0),1,length)
        matrix[:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        fofe_code = torch.bmm(matrix,x).squeeze(-2)
        return fofe_code

class fofe_dual(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe_dual, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        length = x.size(-2)
        matrix_s = x.new_empty(x.size(0),1,length)
        matrix_s[:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
        matrix_l = x.new_empty(x.size(0),1,length)
        matrix_l[:,].copy_(torch.pow(self.alpha-0.4,torch.linspace(length-1,0,length)))
        short_fofe = torch.bmm(matrix_s,x).squeeze(-2)
        long_fofe = torch.bmm(matrix_l,x).squeeze(-2)
        fofe_code = torch.cat([short_fofe, long_fofe], dim=-1)
        return fofe_code


class fofe_flex(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe_flex, self).__init__()
        self.alpha = Parameter(torch.ones(1)*alpha)
        self.alpha.requires_grad_(True)
        
    def forward(self, x):
        length = x.size(-2)
        #import pdb; pdb.set_trace()
        matrix = torch.pow(self.alpha,torch.linspace(length-1,0,length).cuda()).unsqueeze(0)
        fofe_code = matrix.matmul(x).squeeze(-2)
        return fofe_code


class fofe_flex_dual(nn.Module):
    def __init__(self, channels, alpha_l, alpha_h): 
        super(fofe_flex_dual, self).__init__()
        self.alpha_l = Parameter(torch.ones(1)*alpha_l)
        self.alpha_h = Parameter(torch.ones(1)*alpha_h)
        self.alpha_l.requires_grad_(True)
        self.alpha_h.requires_grad_(True)
        
    def forward(self, x):
        length = x.size(-2)
        #import pdb; pdb.set_trace()
        matrix_l = torch.pow(self.alpha_l, torch.linspace(length-1,0,length).cuda()).unsqueeze(0)
        matrix_h = torch.pow(self.alpha_h, torch.linspace(length-1,0,length).cuda()).unsqueeze(0)
        fofe_l = matrix_l.matmul(x).squeeze(-2)
        fofe_h = matrix_h.matmul(x).squeeze(-2)
        fofe_code = torch.cat([fofe_l, fofe_h], dim=-1)
        return fofe_code


class fofe_flex_dual_filter(nn.Module):
    def __init__(self, inplanes, alpha_l=0.4, alpha_h=0.8, length=3, inverse=False):
        super(fofe_flex_dual_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha_l = Parameter(torch.ones(1)*alpha_l)
        self.alpha_h = Parameter(torch.ones(1)*alpha_h)
        self.alpha_l.requires_grad_(True)
        self.alpha_h.requires_grad_(True)
        self.inverse = inverse

    def forward(self, x):
        fofe_kernel_l = x.new_zeros(x.size(1), 1, self.length)
        fofe_kernel_h = x.new_zeros(x.size(1), 1, self.length)
        if self.inverse:
            fofe_kernel_l[:,:,]=torch.pow(self.alpha_l, torch.range(0, self.length-1).cuda())
            fofe_kernel_h[:,:,]=torch.pow(self.alpha_h, torch.range(0, self.length-1).cuda())
            x = F.pad(x,(0, self.length))
        else :
            fofe_kernel_l[:,:,]=torch.pow(self.alpha_l, torch.linspace(self.length-1, 0, self.length).cuda())
            fofe_kernel_h[:,:,]=torch.pow(self.alpha_h, torch.linspace(self.length-1, 0, self.length).cuda())
            x = F.pad(x,(self.length, 0))
        fofe_l = F.conv1d(x, fofe_kernel_l, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        fofe_h = F.conv1d(x, fofe_kernel_h, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        fofe_code = torch.cat([fofe_l, fofe_h], dim=1)

        return fofe_code


class fofe_flex_filter(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_flex_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = Parameter(torch.ones(1)*alpha)
        self.alpha.requires_grad_(True)
        self.inverse = inverse

    def forward(self, x):
        #if self.alpha == 1 or self.alpha == 0 :
        #    self.alpha = 0.9
        fofe_kernel = x.new_zeros(x.size(1), 1, self.length)
        if self.inverse:
            fofe_kernel[:,:,]=torch.pow(self.alpha, torch.range(0, self.length-1).cuda())
            x = F.pad(x,(0, self.length))
        else :
            fofe_kernel[:,:,]=torch.pow(self.alpha, torch.linspace(self.length-1, 0, self.length).cuda())
            x = F.pad(x,(self.length, 0))
        x = F.conv1d(x, fofe_kernel, bias=None, stride=1, 
                        padding=0, groups=self.channels)

        return x


class Simility(nn.Module):
    def __init__(self, planes):
        super(Simility, self).__init__()
        self.W = nn.Conv2d(planes*3, 1, 1, 1, bias=False)
        self.W.weight.data = nn.init.kaiming_normal_(self.W.weight.data)

    def forward(self, doc, query):
        d_length = doc.size(-1)
        q_length = query.size(-1)
        d_matrix = []
        q_matrix = []
        a_matrix = []
        
        for i in range(q_length):
            d_matrix.append(doc.unsqueeze(-2))
        for j in range(d_length):
            q_matrix.append(query.unsqueeze(-1))
        
        d_matrix = torch.cat(d_matrix, dim=-2)
        q_matrix = torch.cat(q_matrix, dim=-1)
        s_matrix = d_matrix.mul(q_matrix)

        a_matrix.append(d_matrix)
        a_matrix.append(q_matrix)
        a_matrix.append(s_matrix)
        a_matrix = torch.cat(a_matrix, dim=1)
        simility = self.W(a_matrix).squeeze(1)
        

        return simility


class Attention(nn.Module):
    def __init__(self, planes, q2c=True, bidirection=False):
        super(Attention, self).__init__()
        self.simility = Simility(planes)
        self.q2c = q2c
        self.bidirection = bidirection
    
    def forward(self, doc, query):
        simility = self.simility(doc, query)
        s1_t = F.softmax(simility,dim=-2).transpose(-1,-2)
        c2q_att = torch.bmm(s1_t, query.transpose(-1,-2)).transpose(-1,-2) #batchsize x d x n
        s2 = F.softmax(simility, dim=-1)
        q2c_att = torch.bmm(s1_t, torch.bmm(s2, doc.transpose(-1,-2))).transpose(-1,-2) #batchsize x d x n

        output = []
        output.append(doc)
        output.append(c2q_att)
        output.append(doc.mul(c2q_att))
        output.append(doc.mul(q2c_att))
        output = torch.cat(output, dim=1)

        return output



