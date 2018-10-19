import torch
import argparse
from collections import Counter


class AverageMeter(object):
    """Keep exponential weighted averages."""
    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0
        self.value = 0
        self.t = 0

    def state_dict(self):
        return vars(self)

    def load(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)

    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.value = self.moment / (1 - self.beta ** self.t)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def tri_num(n):
    '''
        Calculate Triangular Number + n;
        Triangular Number = 1+2+...+n
    '''
    return round(n + (n * (n - 1) / 2))


def count_num_substring(arg_max_substring_len, arg_string_len):
    '''
        Count number of substring of length <= 'max_substring_len' with in string of 'length string_len';
    '''
    if (arg_max_substring_len < arg_string_len):
        max_substring_len = arg_max_substring_len
        n_substring = tri_num(max_substring_len) + \
                        ( (arg_string_len - max_substring_len) * max_substring_len )
    else:
        max_substring_len = arg_string_len
        n_substring = tri_num(max_substring_len)
    return n_substring, max_substring_len


#TODO: SHOULDN'T USE POS TAGGER FOR SENTENCE SPLITTING, USE SENTENCE BOUNDARY DETECTOR.
def find_sentence_boundary_from_pos_tagger(doc_pos, get_stacked_batch=False):
    """
        doc_pos = document/context POS tags; [batch * len_d * n_pos_types]
        """
    doc_len = doc_pos.size(1)
    batch_size = doc_pos.size(0)
    sent_boundary_pos_type_idx = 7
    # 1. get sent_boundaries by pos_tagger
    sent_boundaries = doc_pos[:,:,sent_boundary_pos_type_idx].nonzero()
    
    # 2. check all End Of Doc are included in sentence boundaries list
    eod_sent_idx = doc_len-1
    
    # 2.1 count num of missing_sent_boundary
    if sent_boundaries.size(0) == 0:
        missing_sent_boundary = batch_size
    else:
        n_eod_in_curr_sent_boundaries = (sent_boundaries[:,1]==eod_sent_idx).nonzero().size(0)
        missing_sent_boundary = batch_size - n_eod_in_curr_sent_boundaries
    
    # 2.2 if no missing sent boundary, return sent_boundaries
    #     else insert the missing sent boundary into sent_boundaries
    if missing_sent_boundary == 0:
        return sent_boundaries
    else:
        list_sent_boundaries_by_batch = []
        base_idx = 0
        for i in range(batch_size):
            # 2.2.1 get all sent boundaries in curr batch
            if sent_boundaries.size(0) == 0:
                sent_idx_of_batch_i = sent_boundaries.new_empty(0)
            else:
                sent_idx_of_batch_i = (sent_boundaries[:,0]==i).nonzero().squeeze(-1)
            
            # 2.2.2 if no sent boundary at all in curr batch, make End Of Doc as the sole sent boundary
            #       else collect all sent boundaries in curr batch and (if necessary) append End Of Doc to it.
            if sent_idx_of_batch_i.size(0) <= 0:
                sent_boundaries_in_batch_i = sent_boundaries.new_tensor([[i, eod_sent_idx]])
            else:
                sent_boundaries_in_batch_i = sent_boundaries.index_select(0, sent_idx_of_batch_i)
                if sent_boundaries_in_batch_i[-1,1].item() != eod_sent_idx:
                    additonal_sent_boundary = sent_boundaries.new_tensor([[i, eod_sent_idx]])
                    sent_boundaries_in_batch_i = torch.cat((sent_boundaries_in_batch_i,
                                                            additonal_sent_boundary), dim=0)
            list_sent_boundaries_by_batch.append(sent_boundaries_in_batch_i)
        # 2.2.3 concat sent boundaries of all batches
        new_sent_boundaries = torch.cat(list_sent_boundaries_by_batch, dim=0)
        
        # 2.2.4 check if there still missing sent boundary
        if (new_sent_boundaries[:,1]==eod_sent_idx).nonzero().size(0) != batch_size:
            import pdb; pdb.set_trace()
        return new_sent_boundaries


def f1_score_word_lvl(pred_tensor, ans_tensor):
    '''
        pred_tensor = 1D tensor representing prediction's word_vector.
        ans_tensor = 1D tensor representing target's word_vector.
    '''
    pred_word_vector = [pred_tensor[i].item() for i in range(pred_tensor.size(0))]
    ans_word_vector = [ans_tensor[i].item() for i in range(ans_tensor.size(0))]
    common = Counter(pred_word_vector) & Counter(ans_word_vector)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1. * num_same / len(pred_word_vector)
    recall = 1. * num_same / len(ans_word_vector)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

