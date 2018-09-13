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
    def extra_repr(self):
        return 'inplanes={channels}, alpha={alpha}, ' \
                'length={length}, inverse={inverse}'.format(**self.__dict__)


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
    def __init__(self, channels, alpha, inverse=False): 
        super(fofe, self).__init__()
        self.alpha = alpha
        self.inverse = inverse
        
    def forward(self, x, x_mask):
        length = x.size(-2)
        exponent = x.new_empty(x.size(0),1,length)
        if self.inverse :
            exponent.copy_(torch.range(0, length-1))
        else:
            exponent.copy_(torch.linspace(length-1,0,length))
            exponent.add_( x_mask.sum(1).unsqueeze(-1).unsqueeze(-1).mul(-1))   
        matrix = torch.pow(self.alpha, exponent).mul(1-x_mask.unsqueeze(1))
        fofe_code = torch.bmm(matrix,x).transpose(-1,-2)

        return fofe_code
    
    def extra_repr(self):
        return 'alpha={alpha}, inverse={inverse}'.format(**self.__dict__)

# Sed's version of fofe, need merge, different in shape
class sed_fofe(nn.Module):
    def __init__(self, channels, alpha, inverse=False):
        super(sed_fofe, self).__init__()
        self.alpha = alpha
        self.inverse = inverse
    
    def forward(self, x_input, x_mask):
        length = x_input.size(-2)
        
        # Construct Alphas Buffer
        alpha_buffer = x_input.new_empty(x_input.size(0),1,length)
        if self.inverse:
            alpha_buffer[:,].copy_(torch.pow(self.alpha,torch.range(0, length-1)))
        else:
            alpha_buffer[:,].copy_(torch.pow(self.alpha,torch.linspace(length-1,0,length)))
            alpha_buffer_padding_rm = torch.pow(self.alpha,-1*torch.sum(x_mask,dim=1).float()).unsqueeze(-1).unsqueeze(-1)
            alpha_buffer.copy_(torch.bmm(alpha_buffer_padding_rm, alpha_buffer))

        # Adjust Alpha Buffer to accommondate the padding
        rev_x_mask = (1 - x_mask).unsqueeze(1).float()
        alpha_buffer.copy_(torch.mul(alpha_buffer, rev_x_mask))
        
        # Compute FOFE Code
        fofe_code = torch.bmm(alpha_buffer,x_input).squeeze(-2)
        
        return fofe_code


class fofe_flex(nn.Module):
    def __init__(self, inplanes, alpha): 
        super(fofe_flex, self).__init__()
        self.alpha = Parameter(torch.ones(1)*alpha)
        self.alpha.requires_grad_(True)
        
    def forward(self, x):
        length = x.size(-2)
        #import pdb; pdb.set_trace()
        matrix = torch.pow(self.alpha,torch.linspace(length-1,0,length).cuda()).unsqueeze(0)
        fofe_code = matrix.matmul(x).squeeze(-2)
        return fofe_code

    def extra_repr(self):
        return 'inplanes={inplanes}'.format(**self.__dict__)


class fofe_flex_all(nn.Module):
    def __init__(self, channels, alpha, inverse=False): 
        super(fofe_flex_all, self).__init__()
        self.channels = channels
        self.inverse = inverse
        self.init_alpha = alpha
        self.alpha = Parameter(torch.ones(channels, 1)*alpha)
        self.alpha.requires_grad_(True)
        
    def forward(self, x, x_mask):
        length = x.size(-2)
        exponent = x.new_empty(x.size(0),1,length)
        if self.inverse :
            exponent.copy_(torch.range(0, length-1))
        else:
            exponent.copy_(torch.linspace(length-1,0,length))
            exponent.add_( x_mask.sum(1).unsqueeze(-1).unsqueeze(-1).mul(-1))   
        matrix = torch.pow(self.alpha, exponent).mul(1-x_mask.unsqueeze(1))
        fofe_code = x.transpose(-1,-2).mul(matrix).sum(-1).unsqueeze(-1)
        return fofe_code
    
    def extra_repr(self):
        return 'inplanes={channels}, alpha={init_alpha}, inverse={inverse}'.format(**self.__dict__)



class fofe_flex_all_conv(nn.Module):
    def __init__(self, inplanes, planes, alpha): 
        super(fofe_flex_all_conv, self).__init__()
        self.fofe = fofe_flex_all(inplanes, alpha)
        self.conv = nn.Sequential(
            nn.Conv1d(inplanes, planes, 1, 1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, x_mask):
        fofe_code = self.fofe(x, x_mask)
        out = self.conv(fofe_code)
        return out


class fofe_flex_all_filter(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_flex_all_filter, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = Parameter(torch.ones(inplanes, 1)*alpha)
        self.alpha.requires_grad_(True)
        self.inverse = inverse

    def forward(self, x):
        if self.inverse:
            fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.range(0, self.length-1))).unsqueeze(1)
            x = F.pad(x,(0, self.length))
        else :
            fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.linspace(self.length-1, 0, self.length))).unsqueeze(1)
            x = F.pad(x,(self.length, 0))
        x = F.conv1d(x, fofe_kernel, bias=None, stride=1, 
                        padding=0, groups=self.channels)
        
        return x

    def extra_repr(self):
        return 'inplanes={channels}, length={length}， ' \
            'inverse={inverse}'.format(**self.__dict__)


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
            fofe_kernel[:,:,]=torch.pow(self.alpha, x.new_tensor(torch.range(0, self.length-1)))
            x = F.pad(x,(0, self.length))
        else :
            fofe_kernel[:,:,]=torch.pow(self.alpha, x.new_tensor(torch.linspace(self.length-1, 0, self.length)))
            x = F.pad(x,(self.length, 0))
        x = F.conv1d(x, fofe_kernel, bias=None, stride=1, 
                        padding=0, groups=self.channels)

        return x

    def extra_repr(self):
        return 'inplanes={inplanes}, length={length}， ' \
            'inverse={inverse}'.format(**self.__dict__)


class fofe_encoder(nn.Module):
    def __init__(self, emb_dim, fofe_alpha, fofe_max_length):
        super(fofe_encoder, self).__init__()
        self.forward_filter = []
        self.inverse_filter = []
        for i in range(fofe_max_length):
            self.forward_filter.append(fofe_filter(emb_dim, fofe_alpha, i+1))
            self.inverse_filter.append(fofe_filter(emb_dim, fofe_alpha, i+1, inverse=True))

        self.forward_filter = nn.ModuleList(self.forward_filter)
        self.inverse_filter = nn.ModuleList(self.inverse_filter)
    
    def forward(self, x, max_len):
        forward_fofe = []
        inverse_fofe = []
#        for forward_filter in self.forward_filter:
#            forward_fofe.append(forward_filter(x).unsqueeze(-2))
#        for inverse_filter in self.inverse_filter:
#            inverse_fofe.append(inverse_filter(x).unsqueeze(-2))
        for i in range(max_len):
            forward_fofe.append(self.forward_filter[i](x).unsqueeze(-2))
            inverse_fofe.append(self.inverse_filter[i](x).unsqueeze(-2))
            
        forward_fofe.append(self.forward_filter[-1](x).unsqueeze(-2))
        inverse_fofe.append(self.inverse_filter[-1](x).unsqueeze(-2))
        
        forward_fofe = torch.cat(forward_fofe, dim=-2)
        inverse_fofe = torch.cat(inverse_fofe, dim=-2)
        
        return forward_fofe, inverse_fofe


class fofe_encoder_conv(fofe_encoder):
    def __init__(self, inplanes, planes, fofe_alpha, fofe_max_length):
        super(fofe_encoder_conv, self).__init__(inplanes, fofe_alpha, fofe_max_length)
        self.forward_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.inverse_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        forward_fofe, inverse_fofe = self.fofe(x)
        forward = self.forward_conv(forward_fofe)
        inverse = self.inverse_conv(inverse_fofe)

        return forward, inverse


class fofe_flex_all_encoder(nn.Module):
    def __init__(self, inplanes, alpha=0.8, length=3, inverse=False):
        super(fofe_flex_all_encoder, self).__init__()
        self.length = length
        self.channels = inplanes
        self.alpha = Parameter(torch.ones(inplanes, 1)*alpha)
        self.alpha.requires_grad_(True)
        self.inverse = inverse

    def forward(self, x, max_len):
        out = []
        for i in range(max_len):
            if self.inverse:
                fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.range(0, i-1))).unsqueeze(1)
                out.append(F.conv1d(F.pad(x,(0, i)), fofe_kernel, bias=None, stride=1, 
                                padding=0, groups=self.channels).unsqueeze(-2))
            else:
                fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.linspace(i-1, 0, i))).unsqueeze(1)
                out.append(F.conv1d(F.pad(x,(i, 0)), fofe_kernel, bias=None, stride=1, 
                                padding=0, groups=self.channels).unsqueeze(-2))        
        if self.inverse:
            fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.range(0, 64-1))).unsqueeze(1)
            out.append(F.conv1d(F.pad(x,(0, i)), fofe_kernel, bias=None, stride=1, 
                            padding=0, groups=self.channels).unsqueeze(-2))
        else :
            fofe_kernel = torch.pow(self.alpha, x.new_tensor(torch.linspace(64-1, 0, 64))).unsqueeze(1)
            out.append(F.conv1d(F.pad(x,(64, 0)), fofe_kernel, bias=None, stride=1, 
                            padding=0, groups=self.channels).unsqueeze(-2))

        out = torch.cat(out, dim=-2)
        return out


class fofe_encoder_conv(fofe_encoder):
    def __init__(self, inplanes, planes, fofe_alpha_l, fofe_alpha_h, fofe_max_length):
        super(fofe_encoder_conv, self).__init__(inplanes, fofe_alpha_l, fofe_alpha_h, fofe_max_length)
        self.forward_conv = nn.Sequential(
            nn.Conv2d(inplanes*2, planes*2, 1, 1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(inplace=True)
        )
        self.inverse_conv = nn.Sequential(
            nn.Conv2d(inplanes*2, planes*2, 1, 1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        forward_fofe, inverse_fofe = self.fofe(x)
        forward = self.forward_conv(forward_fofe)
        inverse = self.inverse_conv(inverse_fofe)

        return forward, inverse


class fofe_multi(nn.Module):
    def __init__(self, filter, emb_dim, fofe_alpha):
        super(fofe_multi, self).__init__()
        self.fofe_alpha = fofe_alpha
        self.emb_dim = emb_dim
        self.fofe_matrix = []
        for alpha in fofe_alpha:
            self.fofe_matrix.append(filter(emb_dim, alpha, inverse=False))
            self.fofe_matrix.append(filter(emb_dim, alpha, inverse=True))

        self.fofe_matrix = nn.ModuleList(self.fofe_matrix)
    
    def forward(self, x, x_mask):
        fofe_code = []
        for fofe_matrix in self.fofe_matrix:
            fofe_code.append(fofe_matrix(x, x_mask))
        fofe_code = torch.cat(fofe_code, dim=1)
        
        return fofe_code


class fofe_multi_filter(nn.Module):
    def __init__(self, filter, emb_dim, fofe_alpha, fofe_length, inverse=False):
        super(fofe_multi_filter, self).__init__()
        self.emb_dim = emb_dim
        self.fofe_alpha = fofe_alpha
        self.fofe_length = fofe_length
        self.inverse = inverse
        self.filter = []
        for alpha in fofe_alpha:
            self.filter.append(filter(emb_dim, alpha, fofe_length, inverse))

        self.filter = nn.ModuleList(self.filter)
    
    def forward(self, x):
        fofe_code = []
        for filter in self.filter:
            fofe_code.append(filter(x))
        fofe_code = torch.cat(fofe_code, dim=1)
        return fofe_code


class fofe_multi_encoder(fofe_encoder):
    def __init__(self, filter, emb_dim, fofe_alpha, fofe_max_length):
        super(fofe_encoder, self).__init__()
        self.forward_filter = []
        self.inverse_filter = []
        for i in range(fofe_max_length):
            self.forward_filter.append(fofe_multi_filter(filter, emb_dim, fofe_alpha, i+1))
            self.inverse_filter.append(fofe_multi_filter(filter, emb_dim, fofe_alpha, i+1, inverse=True))

        self.forward_filter = nn.ModuleList(self.forward_filter)
        self.inverse_filter = nn.ModuleList(self.inverse_filter)


class fofe_tricontext(nn.Module):
    def __init__(self, embedding_dim, alpha, cand_len_limit=10, doc_len_limit=809, has_lr_ctx_cand_incl=True, has_lr_ctx_cand_excl=True, inverse=False):
        super(fofe_tricontext, self).__init__()
        self.alpha = alpha
        self.cand_len_limit = cand_len_limit
        self.doc_len_limit = doc_len_limit
        self.has_lr_ctx_cand_incl = has_lr_ctx_cand_incl
        self.has_lr_ctx_cand_excl = has_lr_ctx_cand_excl
        self.inverse_cand_fofe = inverse
        
        # Construct Base Alpha Buffer for full doc length.
        self._full_base_block_alpha = torch.zeros(self.doc_len_limit, self.doc_len_limit)
        for i in range(1, self.doc_len_limit+1):
            powers = torch.linspace(i-1,i-self.doc_len_limit,self.doc_len_limit).abs()
            self._full_base_block_alpha[i-1,:].copy_(torch.pow(self.alpha,powers))
        
        # NOTED: self._full_base_tril_inv_alpha is only needed, when self.inverse_cand_fofe == True.
        self._full_base_tril_inv_alpha = torch.zeros(self.doc_len_limit, self.doc_len_limit)
        if (self.inverse_cand_fofe):
            powers = torch.range(0, self.doc_len_limit-1)
            self._full_base_tril_inv_alpha.copy_(torch.pow(self.alpha, powers)\
                                                 .expand(self.doc_len_limit, self.doc_len_limit)\
                                                 .tril())

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

    def get_contexts_alpha_buffers(self, x_input, test_mode=False):
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
        _base_triu_alpha = x_input.new_zeros(doc_len,doc_len)
        _base_tril_alpha = x_input.new_zeros(doc_len,doc_len)
        _base_tril_inv_alpha = x_input.new_zeros(doc_len,doc_len)
        _base_triu_alpha.copy_(self._full_base_block_alpha.triu()[:doc_len,:doc_len])
        _base_tril_alpha.copy_(self._full_base_block_alpha.tril()[:doc_len,:doc_len])
        _base_tril_inv_alpha.copy_(self._full_base_tril_inv_alpha[:doc_len,:doc_len])
        
        #calculate number of candidate (aka n_cand); and set max_cand_len = min(doc_len, self.cand_len_limit)
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
                if (self.inverse_cand_fofe):
                    context_alpha_buffers[0][start_idx:end_idx,i:max_cand_len+i].copy_(_base_tril_inv_alpha[:max_cand_len,:max_cand_len])
                else:
                    context_alpha_buffers[0][start_idx:end_idx,i:max_cand_len+i].copy_(_base_tril_alpha[i:max_cand_len+i,i:max_cand_len+i])

                # Left Context (Candidate included) alpha buffer
                if (self.has_lr_ctx_cand_incl):
                    context_alpha_buffers[2][start_idx:end_idx,:max_cand_len+i].copy_(_base_tril_alpha[i:max_cand_len+i,:max_cand_len+i])
                
                # Right Context (Candidate excluded) alpha buffer
                if (self.has_lr_ctx_cand_excl):
                    context_alpha_buffers[3][start_idx:end_idx,1:].copy_(_base_triu_alpha[i:max_cand_len+i,:-1])
        
            else:
                rev_i = doc_len-i-1
                base_idx = (doc_len - max_cand_len) * max_cand_len
                start_idx = base_idx + tri_num(max_cand_len) - tri_num(rev_i+1)
                end_idx = base_idx + tri_num(max_cand_len) - tri_num(rev_i)
                
                # Candidate Context alphas buffer
                if (self.inverse_cand_fofe):
                    context_alpha_buffers[0][start_idx:end_idx,i:doc_len].copy_(_base_tril_inv_alpha[:doc_len-i,:doc_len-i])
                else:
                    context_alpha_buffers[0][start_idx:end_idx,i:doc_len].copy_(_base_tril_alpha[i:doc_len,i:doc_len])

                # Left Context (Candidate included) alpha buffer
                if (self.has_lr_ctx_cand_incl):
                    context_alpha_buffers[2][start_idx:end_idx,:].copy_(_base_tril_alpha[i:doc_len,:doc_len])
                
                # Right Context (Candidate excluded) alpha buffer
                if (self.has_lr_ctx_cand_excl):
                    context_alpha_buffers[3][start_idx:end_idx,1:].copy_(_base_triu_alpha[i:doc_len,:-1])
            
            # Left Context (Candidate excluded) alpha buffer
            # i > 0 SINCE: left context doesn't exist for BOS (begin of sentence) candidates,
            #          SO: leave the values as zero (when i == 0).
            if (i > 0 and self.has_lr_ctx_cand_excl):
                context_alpha_buffers[1][start_idx:end_idx,:].copy_(_base_tril_alpha[i-1,:doc_len]\
                                                                    .expand(end_idx-start_idx,doc_len))

            # Right Context (Candidate included) alpha buffer
            if (self.has_lr_ctx_cand_incl):
                context_alpha_buffers[4][start_idx:end_idx,:].copy_(_base_triu_alpha[i,:doc_len]\
                                                                    .expand(end_idx-start_idx,doc_len))

            # Candidate Positions within Doc
            if (test_mode):
                cands_pos[start_idx:end_idx, 0] = i
                cands_pos[start_idx:end_idx, 1].copy_(torch.range(i, i+end_idx-start_idx-1))
        
        if (test_mode):
            return context_alpha_buffers, cands_pos
        else:
            return context_alpha_buffers

    
    def forward(self, x_input, x_mask, test_mode=False):
        #TODO @SED: Expand the cand_len_limit to match len of ans (if len of ans > cand_len_limit)
        batch_size = x_input.size(0)
        length = min(x_input.size(1), self.doc_len_limit)
        embedding_dim = x_input.size(2)
        
        if (test_mode):
            context_alpha_buffer, cands_pos = self.get_contexts_alpha_buffers(x_input, test_mode)
        else:
            context_alpha_buffer = self.get_contexts_alpha_buffers(x_input, test_mode)
        
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
                                              _batchwise_fofe_codes[2],
                                              _batchwise_fofe_codes[3],
                                              _batchwise_fofe_codes[4]], dim=-1)
        if (test_mode):
            batchwise_cands_pos = cands_pos.unsqueeze(0).expand(batch_size,n_cand, cands_pos.size(-1))
            batchwise_padded_cands = torch.bmm(_batchwise_alpha_buffer[0], x_mask.float().unsqueeze(-1)) > 0
            return batchwise_fofe_codes, batchwise_cands_pos, batchwise_padded_cands
        else:
            return batchwise_fofe_codes


class bidirect_fofe_tricontext(nn.Module):
    def __init__(self, embedding_dim, alpha, cand_len_limit=10, doc_len_limit=46, has_lr_ctx_cand_incl=True, has_lr_ctx_cand_excl=True):
        super(bidirect_fofe_tricontext, self).__init__()
        self.alpha = alpha
        self.cand_len_limit = cand_len_limit
        self.doc_len_limit = doc_len_limit
        self.has_lr_ctx_cand_incl = has_lr_ctx_cand_incl
        self.has_lr_ctx_cand_excl = has_lr_ctx_cand_excl
        self.forward_fofe = fofe_tricontext(embedding_dim,
                                            self.alpha,
                                            self.cand_len_limit,
                                            doc_len_limit,
                                            has_lr_ctx_cand_incl=False,
                                            has_lr_ctx_cand_excl=False,
                                            inverse=False)
        self.backward_fofe = fofe_tricontext(embedding_dim,
                                             alpha,
                                             cand_len_limit,
                                             doc_len_limit,
                                             has_lr_ctx_cand_incl=self.has_lr_ctx_cand_incl,
                                             has_lr_ctx_cand_excl=self.has_lr_ctx_cand_excl,
                                             inverse=True)

    def forward(self, x_input, x_mask, test_mode=False):
        if (test_mode):
            backward_fofe_code, cands_pos, padded_cands = self.backward_fofe(x_input, x_mask, test_mode)

            forward_fofe_code, _, _ = self.forward_fofe(x_input, x_mask, test_mode)
            fofe_code = torch.cat([forward_fofe_code,backward_fofe_code], dim=-1)
            return fofe_code, cands_pos, padded_cands
        else:
            backward_fofe_code = self.backward_fofe(x_input, x_mask, test_mode)
            forward_fofe_code = self.forward_fofe(x_input, x_mask, test_mode)
            fofe_code = torch.cat([forward_fofe_code,backward_fofe_code], dim=-1)
            return fofe_code


class bidirect_fofe(nn.Module):
    def __init__(self, channels, alpha):
        super(bidirect_fofe, self).__init__()
        self.forward_fofe = sed_fofe(channels, alpha, inverse=False)
        self.backward_fofe = sed_fofe(channels, alpha, inverse=True)

    def forward(self, x_input, x_mask):
        forward_fofe_code = self.forward_fofe(x_input, x_mask)
        backward_fofe_code = self.backward_fofe(x_input, x_mask)
        fofe_code = torch.cat([forward_fofe_code,backward_fofe_code], dim=-1)
        return fofe_code


class bidirect_fofe_multi_tricontext(nn.Module):
    # bidirect_fofe_multi_tricontext encoder design for doc input.
    def __init__(self, fofe_alphas, doc_input_size, cand_len_limit, doc_len_limit, contexts_incl_cand, contexts_excl_cand):
        super(bidirect_fofe_multi_tricontext, self).__init__()
        self.fofe_encoders = []
        for _, fofe_alpha in enumerate(fofe_alphas):
            self.fofe_encoders.append(bidirect_fofe_tricontext(doc_input_size,
                                                               fofe_alpha,
                                                               cand_len_limit=cand_len_limit,
                                                               doc_len_limit=doc_len_limit,
                                                               has_lr_ctx_cand_incl=contexts_incl_cand,
                                                               has_lr_ctx_cand_excl=contexts_excl_cand))
        self.fofe_encoders = nn.ModuleList(self.fofe_encoders)
    
    def forward(self, doc_emb, doc_mask, test_mode=False):
        doc_fofe = []
        for d_fofe_encoder in self.fofe_encoders:
            if test_mode:
                _doc_fofe, _cands_ans_pos, _padded_cands_mask = d_fofe_encoder(doc_emb, doc_mask, test_mode)
            else:
                _doc_fofe = d_fofe_encoder(doc_emb, doc_mask, test_mode)
            doc_fofe.append(_doc_fofe)
        doc_fofe = torch.cat(doc_fofe, dim=-1)

        if test_mode:
            return doc_fofe, _cands_ans_pos, _padded_cands_mask
        else: 
            return doc_fofe

        
class bidirect_fofe_multi(nn.Module):
    # bidirect_fofe_multi encoder design for query input.
    def __init__(self, fofe_alphas, query_input_size):
        super(bidirect_fofe_multi, self).__init__()
        self.fofe_encoders = []
        for _, fofe_alpha in enumerate(fofe_alphas):
            self.fofe_encoders.append(bidirect_fofe(query_input_size, fofe_alpha))
        self.fofe_encoders = nn.ModuleList(self.fofe_encoders)

    def forward(self, query_emb, query_mask, batch_size,n_cands_ans):
        query_fofe = []
        for q_fofe_encoder in self.fofe_encoders:
            _query_fofe = q_fofe_encoder(query_emb, query_mask)
            query_embedding_dim = _query_fofe.size(-1)
            query_fofe.append(_query_fofe.unsqueeze(1).expand(batch_size,n_cands_ans,query_embedding_dim))
        query_fofe = torch.cat(query_fofe, dim=-1)
        return query_fofe

