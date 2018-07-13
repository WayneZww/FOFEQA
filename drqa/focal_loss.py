import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    # Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    # Loss(log_softmax(x), class) = - \alpha (1-exp(log_softmax(x))[class])^gamma \log_softmax(x)[class]
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        class_mask = inputs.new(inputs.shape).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        alpha = inputs.new_tensor(self.alpha[ids.view(-1)])        
        log_p = (inputs*class_mask).sum(1).view(-1,1)
        batch_loss = -alpha*(torch.pow((torch.exp(log_p)), self.gamma))*log_p 
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    

class FocalLoss1d(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=5, size_average=True):
        super(FocalLoss1d, self).__init__()
        self.FL = FocalLoss(class_num, alpha, gamma, size_average)

    def forward(self, inputs, targets):
        N, C, W = inputs.size()
        reshaped_inputs = inputs.transpose(-1, -2).contiguous().view(N*W, C)
        reshaped_targets = targets.contiguous().view(N*W)
        loss = self.FL(reshaped_inputs, reshaped_targets)
        return loss