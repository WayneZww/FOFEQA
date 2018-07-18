# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from .utils import AverageMeter
from .fofe_reader import FOFEReader

# Modification:
#   - change the logger name
#   - save & load "state_dict"s of optimizer and loss meter
#   - save all random seeds
#   - change the dimension of inputs (for POS and NER features)
#   - remove "reset parameters" and use a gradient hook for gradient masking
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

logger = logging.getLogger(__name__)


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.device = torch.cuda.current_device() if opt['cuda'] else torch.device('cpu')
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()
        if state_dict:
            self.train_loss.load(state_dict['loss'])

        # Building network.
        self.network = FOFEReader(opt, embedding=embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])
        self.network.to(self.device)

        # Building optimizer.
        self.opt_state_dict = state_dict['optimizer'] if state_dict else None
        self.build_optimizer()

    def build_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.opt['learning_rate'],
                                       momentum=self.opt['momentum'],
                                       weight_decay=self.opt['weight_decay'])
        elif self.opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.opt['optimizer'])
        if self.opt_state_dict:
            self.optimizer.load_state_dict(self.opt_state_dict)

    def update(self, ex):
        # Train mode
        self.network.train()
        
        if self.opt['cuda']:
            inputs = [Variable(e.cuda()) for e in ex[:9]]
        else:
            inputs = [Variable(e) for e in ex[:9]]
        
        # Run forward
        loss = self.network(*inputs)
        self.train_loss.update(loss.item())
        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients it helps converges
        #torch.nn.utils.clip_grad_norm_(self.network.parameters(),
        #                              self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:7]]
        else:
            inputs = [Variable(e) for e in ex[:7]]

        #----------------------------------------------------------------------------
        #TODO @SED: CHECK THE PREDICTION
        # Run forward
        with torch.no_grad():
            predict_s_idx, predict_e_idx = self.network(*inputs)

        # Get text prediction
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        for i in range(len(predict_s_idx)):
            try:
                s_idx = predict_s_idx[i]
                e_idx = predict_e_idx[i]
                #if s_idx < 0 or e_idx < 0:
                #    predictions.append("<UNK>")
                #else:
                #    s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                #    predictions.append(text[i][s_offset:e_offset])
                s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                predictions.append(text[i][s_offset:e_offset])
            except Exception as e:
                #TODO @SED: CURRENT HAVE PROBLEM OF SOMETIME PREDICTING AN OUT OF BOUND, WHEN BATCHSIZE > 1
                #           -Suspect that it predict the padding, since we test without training.
                print("--------------------------------------------------------------")
                print(e)
                import pdb;pdb.set_trace()
                print("s_idx:", s_idx)
                print("e_idx:", e_idx)
                print("spans[i]:", spans[i])
                print("spans:", spans)
                print("text", text)
                predictions.append("<PAD>")
        #----------------------------------------------------------------------------
        return predictions

    def save(self, filename, epoch, scores):
        em, f1, best_eval = scores
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates,
                'loss': self.train_loss.state_dict()
            },
            'config': self.opt,
            'epoch': epoch,
            'em': em,
            'f1': f1,
            'best_eval': best_eval,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')
