import argparse


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
