# Modification:
#  -change to support fofe_nn
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from .fofe_modules import fofe_conv1d, fofe_linear, fofe_block, fofe_res_att_block, fofe_res_conv_block, reg_loss
from .fofe_net import FOFENet, FOFENet_Biatt, FOFENet_Biatt_ASPP, FOFENet_Biatt_Selfatt_ASPP


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
        self.reg_loss = None
        if self.opt['regloss_ratio'] > 0:
            self.reg_loss = reg_loss(opt['regloss_sigma'])

        if opt['block'] == 'fofe_res_att_block':
            block = fofe_res_att_block
        elif opt['block'] == 'fofe_res_conv_block':
            block = fofe_res_conv_block
        else:
            raise Exception('Block architecture undefined!')
        
        net_config = [block, opt['embedding_dim'], 
                                opt['planes'],
                                opt['fofe_alpha'],
                                opt['fofe_max_length']]
        
        if opt['encoder'] == 'fofe' :
            self.fofe_nn = FOFENet(*net_config)
        elif opt['encoder'] == 'fofe_biatt' :
            self.fofe_nn = FOFENet_Biatt(*net_config)
        elif opt['encoder'] == 'fofe_biatt_aspp' :
            self.fofe_nn = FOFENet_Biatt_ASPP(*net_config)
        elif opt['encoder'] == 'fofe_biatt_nonlocal_aspp' :
            self.fofe_nn = FOFENet_Biatt_Selfatt_ASPP(*net_config)
        else :
            raise Exception('Architecture undefined!')
        #print(self.fofe_nn)
        
    def forward(self, doc, doc_f, doc_pos, doc_ner, doc_mask, 
                query, query_mask, target_s=None, target_e=None):
        """Inputs:
        doc = document word indices             [batch * len_d]
        doc_f = document word features indices  [batch * len_d * nfeat]
        doc_pos = document POS tags             [batch * len_d]
        doc_ner = document entity tags          [batch * len_d]
        doc_mask = document padding mask        [batch * len_d]
        query = question word indices             [batch * len_q]
        query_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        doc_emb = self.embedding(doc)
        query_emb = self.embedding(query)
        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            doc_emb = nn.functional.dropout(doc_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
            query_emb = nn.functional.dropout(query_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
        
        # Predict start and end positions
        score_s, score_e = self.fofe_nn(query_emb, query_mask, doc_emb, doc_mask)

        # Compute loss and accuracies
        if self.training :
            loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
            if self.opt['regloss_ratio'] > 0:
                reg_loss = self.reg_loss(score_s, target_s) + self.reg_loss(score_e, target_e)
                loss += self.opt['regloss_ratio']*reg_loss
            return loss
        else :
            return score_s, score_e


