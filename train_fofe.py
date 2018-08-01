import re
import os
import sys
import math
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
from drqa.fofe_model import DocReaderModel
from drqa.utils import str2bool
import matplotlib.pyplot as plt
import numpy as np


def main():
    args, log = setup()
    log.info('[Program starts. Loading data...]')

    if args.test_train:
        train, train_y, dev, dev_y, sample_train, sample_train_y, embedding, opt = load_data(vars(args))
    else:
        train, dev, dev_y, sample_train, sample_train_y, embedding, opt = load_data(vars(args))

    log.info(opt)
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(args.model_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        # synchronize random seed
        random.setstate(checkpoint['random_state'])
        torch.random.set_rng_state(checkpoint['torch_state'])
        if args.cuda:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
            log.info('[learning rate reduced by {}]'.format(args.reduce_lr))
            
        # Test  dev and total train
        if args.draw_score:
            test_draw(train, train_y, args, model, log, mode='train')
            return
        sample_em, sample_f1 = test_process(sample_train, sample_train_y, args, model, log, mode='sample_train')
        dev_em, dev_f1 = test_process(dev, dev_y, args, model, log, mode='dev')

        if math.fabs(dev_em - checkpoint['em']) > 1e-3 or math.fabs(dev_f1 - checkpoint['f1']) > 1e-3:
            log.info('Inconsistent: recorded EM: {} F1: {}'.format(checkpoint['em'], checkpoint['f1']))
            log.error('Error loading model: current code is inconsistent with code used to train the previous model.')
            exit(1)
        best_val_score = checkpoint['best_eval']
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1
        best_val_score = 0.0
    
    dev_em_record = []
    dev_f1_record = []
    sample_em_record = []
    sample_f1_record = []
    x_axis = []
    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        if not args.test_only:
            train_process(train, epoch, args, model, log)

        if args.test_only and args.resume:
            break         
        # Test on Dev Set
        dev_em, dev_f1 = test_process(dev, dev_y, args, model, log, mode='dev')
        # Test on sampled train set
        sample_em, sample_f1 = test_process(sample_train, sample_train_y, args, model, log, mode='sample_train')
        # Test on total Train Set
        if args.test_train and epoch % 10 == 0:
            train_em, train_f1 = test_process(train, train_y, args, model, log, mode='train')
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(args.model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch, [dev_em, dev_f1, best_val_score])
            if dev_f1 > best_val_score:
                best_val_score = dev_f1
                copyfile(
                    model_file,
                    os.path.join(args.model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')
        log.debug('\n')

        if args.test_only:
            break
        dev_em_record.append(dev_em) 
        dev_f1_record.append(dev_f1) 
        sample_em_record.append(sample_em)
        sample_f1_record.append(sample_f1)
        x_axis.append(epoch)

        if args.draw_plot:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot()
            x_axis_np = np.asarray(x_axis)
            plt.plot(x_axis_np,np.asarray(dev_em_record),'-',label="Dev EM")
            plt.plot(x_axis_np,np.asarray(dev_f1_record),'-',label="Dev F1")
            plt.plot(x_axis_np,np.asarray(sample_em_record),'-',label="Sampled Train EM")
            plt.plot(x_axis_np,np.asarray(sample_f1_record),'-',label="Sampled Train F1")
            dev_max_idx = np.argmax(np.asarray(dev_f1_record))
            sample_max_idx = np.argmax(np.asarray(sample_f1_record))
            plt.plot(dev_max_idx+1,dev_f1_record[dev_max_idx],'ro')
            plt.annotate('['+str(dev_max_idx+1)+', '+str(dev_f1_record[dev_max_idx])+']',
                         xytext=(dev_max_idx+1,dev_f1_record[dev_max_idx]),
                         xy=(dev_max_idx+1,dev_f1_record[dev_max_idx]))
            plt.plot(sample_max_idx+1,sample_f1_record[sample_max_idx],'bo')
            plt.annotate('['+str(sample_max_idx+1)+', '+str(sample_f1_record[sample_max_idx])+']', 
                         xytext=(sample_max_idx+1,sample_f1_record[sample_max_idx]), 
                         xy=(sample_max_idx+1,sample_f1_record[sample_max_idx]))
            plt.xlabel("Epoch")
            plt.xticks(np.arange(epoch_0, args.epochs+1, 5))
            plt.ylabel("Score")
            plt.yticks(np.arange(20,101,5))
            plt.title("EM & F1 Scores")
            plt.legend()
            plt.savefig(args.model_dir+"/Test.png")
            plt.clf()


def setup():
    parser = argparse.ArgumentParser(
        description='Train a Document Reader model.'
    )
    # system
    parser.add_argument('--log_per_updates', type=int, default=5,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--data_file', default='./data/SQuAD/data.msgpack',
                        help='path to preprocessed data file.')
    parser.add_argument('--meta_file', default='./data/SQuAD/meta.msgpack',
                        help='path to preprocessed data file.')
    parser.add_argument('--model_dir', default='models',
                        help='path to store saved models.')
    parser.add_argument('--save_last_only', action='store_true',
                        help='only save the final models.')
    parser.add_argument('--seed', type=int, default=1013,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    # training
    parser.add_argument('--test_train', action='store_true',
                        help='whether use train set as dev set for debug')
    parser.add_argument('--test_only', action='store_true',
                        help='whether test onlys')
    parser.add_argument('-e', '--epochs', type=int, default=40)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-sn', '--sample_num', type=int, default=256,
                        help='sampling numbers for each doc; \
                        if sn = 0 and nr = 0, will will ignore sampling; \
                        if sn = 0 and nr > 0, will will duplicate up positive sample to match 1-nr ratio.')
    parser.add_argument('-nr', '--neg_ratio', type=float, default=1/2,
                        help='ratio of negtive sample for each doc')
    parser.add_argument('-rs', '--resume', default='best_model.pt',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                        help='use previous model options, ignore the cli and defaults.')
    parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                        help='reduce initial (resumed) learning rate by this factor.')
    parser.add_argument('-op', '--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='only applied to SGD.')
    parser.add_argument('-mm', '--momentum', type=float, default=0,
                        help='only applied to SGD.')
    parser.add_argument('-tp', '--tune_partial', type=int, default=0,
                        help='finetune top-x embeddings.')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--draw_score', action='store_true',
                        help='if true, will draw test score')
    parser.add_argument('--draw_plot', action='store_true',
                        help='if true, will draw test score')

    # model
    parser.add_argument('--contexts_incl_cand', type=str2bool, nargs='?', const=True, default=True,
                        help='Have the Left/Right Contexts that include Candidates')
    parser.add_argument('--contexts_excl_cand', type=str2bool, nargs='?', const=True, default=True,
                        help='Have the Left/Right Contexts that exclude Candidates')
    parser.add_argument('--question_merge', default='self_attn')
    parser.add_argument('--doc_layers', type=int, default=3)
    parser.add_argument('--question_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                        help='use pos tags as a feature.')
    parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=True,
                        help='use named entity tags as a feature.')
    parser.add_argument('--dropout_emb', type=float, default=0.4)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--fofe_alpha', nargs='+', type=float, default='0.8',
                        help='use comma as separator for dual-fofe; (e.g. 0.4,0.8).')
    parser.add_argument('--fofe_max_length', type=int, default=64)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=int, default=2)
    parser.add_argument('--filter', default='fofe',
                        help='Architecture for filter')
    parser.add_argument('--net_arch', default='FOFE_NN',
                        help='Architecture for NN')

    args = parser.parse_args()

    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    if args.resume == 'best_model.pt' and not os.path.exists(os.path.join(args.model_dir, args.resume)):
        # means we're starting fresh
        args.resume = ''

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # setup logger
    class ProgressHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            log_entry = self.format(record)
            if record.message.startswith('> '):
                sys.stdout.write('{}\r'.format(log_entry.rstrip()))
                sys.stdout.flush()
            else:
                sys.stdout.write('{}\n'.format(log_entry))

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
    fh.setLevel(logging.INFO)
    ch = ProgressHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    return args, log


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


def load_data(opt):
    with open(opt['meta_file'], 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    BatchGen.pos_size = opt['pos_size']
    BatchGen.ner_size = opt['ner_size']
    with open(opt['data_file'], 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    
    data['dev'].sort(key=lambda x: len(x[1]))
    dev = [x[:-1] for x in data['dev']]
    dev_y = [x[-1] for x in data['dev']]

    data['train'].sort(key=lambda x: len(x[1]))
    train = [x[:] for x in data['train']]
    train_y = [[x[-3]] for x in data['train']]

    #sample data to test train for each epoch
    sample_idx = range(1, len(train), 10)
    sample_train = [train[i] for i in sample_idx]
    sample_train_y = [train_y[i] for i in sample_idx]

    if opt['test_train']:
        return train, train_y, dev, dev_y, sample_train, sample_train_y, embedding, opt
    else: 
        return train, dev, dev_y, sample_train, sample_train_y, embedding, opt


def train_process(train, epoch, args, model, log):
    batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
    start = datetime.now()
    for i, batch in enumerate(batches):
        model.update(batch)
        if i % args.log_per_updates == 0:
            log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                epoch, model.updates, model.train_loss.value,
                str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))


def test_process(dev, dev_y, args, model, log, mode='dev'):
    test_train=True
    if mode == 'dev':
        log.warning("Evaluating dev set:")
        test_train=False
    elif mode == 'train':
        log.warning("Evaluating total train set:")
    elif mode == 'sample_train':
        log.warning("Evaluating sampled train set:") 

    batches = BatchGen(dev, args.batch_size, test_train=test_train, evaluation=True, gpu=args.cuda)
    predictions = []
    for i, batch in enumerate(batches):
        predictions.extend(model.predict(batch))
        log.debug('> evaluating [{}/{}]'.format(i, len(batches)))
    em, f1 = score(predictions, dev_y)

    if mode == 'dev':
        log.warning("Dev EM: {} F1: {}".format(em, f1))
    elif mode == 'train':
        log.warning("Train EM: {} F1: {}".format(em, f1))
    elif mode == 'sample_train':
        log.warning("Sampled train EM: {} F1: {}".format(em, f1))

    return em, f1


def test_draw(dev, dev_y, args, model, log, mode='dev'):
    batches = BatchGen(dev, args.batch_size, evaluation=True, gpu=args.cuda, draw_score=args.draw_score)
    for i, batch in enumerate(batches):
        model.draw_predict(batch)
        log.debug('> Drawing [{}/{}]'.format(i, len(batches)))



class BatchGen:
    pos_size = None
    ner_size = None

    def __init__(self, data, batch_size, gpu, test_train=False, evaluation=False, draw_score=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu
        self.test_train = test_train 
        self.draw_score = draw_score

        # sort by len
        data = sorted(data, key=lambda x: len(x[1]))
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        # shuffle
        if not evaluation:
            random.shuffle(data)

        self.data = data


    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))

            if self.eval and not self.test_train and not self.draw_score:
                assert len(batch) == 8
            else:
                assert len(batch) == 11

            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            context_ent = torch.Tensor(batch_size, context_len, self.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])
            if not self.eval or self.draw_score:
                y_s = torch.LongTensor(batch[-2])
                y_e = torch.LongTensor(batch[-1])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval and not self.draw_score:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1


if __name__ == '__main__':
    main()

