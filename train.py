'''
This script handles the training process.
'''

import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from CSP.prune import prune
from CSP.pruned_layers import *
from CSP.summary import summary
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from transformer.SubLayers import MultiHeadAttention

__author__ = "Yu-Hsiang Huang"

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, smoothing, finetune=False):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, smoothing=smoothing)

        # Add regularization loss
        if not finetune and opt.spar_str != 0:
            loss = loss.view(1)
            reg_loss = torch.zeros_like(loss).to('cuda')
            for n, m in model.named_modules():
                if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                    reg_loss += m.compute_group_lasso_v2()
            #print("old loss: {}".format(loss))
            loss += reg_loss * opt.spar_str
            #print("new loss: {}".format(loss))
            
        loss.backward()

        if finetune and opt.q != 0:
                # before optimizer.step(), manipulate the gradient
            """                                                                                                                                                                           
            Zero the gradients of the pruned variables.                                                                                                                                   
            """
            for n, m in model.named_modules():
                if isinstance(m, PrunedConv):
                    m.conv.weight.grad = m.conv.weight.grad.float() * m.mask.float()
                if isinstance(m, PrunedLinear):
                    m.linear.weight.grad = m.linear.weight.grad.float() * m.mask.float()
                    #print(m.mask.float())
                    
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt, finetune=False):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    if finetune:
        tlogf = "finetune_train.log"
        vlogf = "finetune_valid.log"
    else:
        tlogf = "train.log"
        vlogf = "valid.log"
        
    log_train_file = os.path.join(opt.output_dir, tlogf)
    log_valid_file = os.path.join(opt.output_dir, vlogf)

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    if finetune:
        epoch = opt.epoch_ft
    else:
        epoch = opt.epoch
        
    #valid_accus = []
    valid_losses = []
    for epoch_i in range(epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing, finetune=finetune)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            if finetune:
                model_name = 'finetune_model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            else:
                model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            if finetune:
                torch.save(checkpoint, os.path.join(model_name, "finetune.pt"))
            else:
                torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            if finetune:
                model_name = 'finetuned_model.chkpt'
            else:
                model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)

def replace_with_pruned(m, name, prune_attention=False, prune_only_attention=False):
    #print(m)                                                                                                                                                                                 
    print("{}, {}".format(name, str(type(m))))
    if type(m) == PrunedConv or type(m) == PrunedLinear:
        return
    if not (prune_attention or prune_only_attention) and type(m) == MultiHeadAttention:
        return

    # HACK: directly replace conv layers of downsamples
    if name == "downsample":
        m[0] = PrunedConv(m[0])

    if not prune_only_attention or type(m) == MultiHeadAttention:
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if type(target_attr) == torch.nn.Conv2d:
                print("Replaced CONV -- ERROR: not expected")
                exit()
                setattr(m, attr_str, PrunedConv(target_attr))
            elif type(target_attr) == torch.nn.Linear:
                print("Replaced Linear")
                setattr(m, attr_str, PrunedLinear(target_attr))
                #replace_with_pruned(m, name, prune_attention)
        # ------------ TODO: REPLACE ATTENTION LAYERS HERE TOO ------------
            
    for n, ch in m.named_children():
        replace_with_pruned(ch, n, prune_attention, prune_only_attention)
            
def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)   # bpe encoded data
    parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-epoch_ft', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    # ED: CSP args
    parser.add_argument('-prune_attention', action='store_true')
    parser.add_argument('-prune_only_attention', action='store_true')
    parser.add_argument('-spar-str', type=float, default=1e-4, help='sparsity reg strength, default=1e-4')
    parser.add_argument('-q', type=float, default=0.75, help='prune threshold, will default to prune-type\'s default if not specified')
    parser.add_argument('-finetune', action='store_true')
    parser.add_argument('-cont', action='store_true', help='continue training from checkpoint')
    parser.add_argument('-model', type=str, help='path to pretrained model')
    
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)

    # ED: replace linear layers with CSP modules                                                                                                                                          
    replace_with_pruned(transformer, "transformer", opt.prune_attention, opt.prune_only_attention)
    for layer in transformer.named_modules():
        print(layer)
    
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    if opt.model != "" and opt.model != None:
        checkpoint = torch.load(opt.model, map_location=device)
        transformer.load_state_dict(checkpoint['model'])
    
    if not opt.finetune:
        if opt.cont:
            assert (opt.model != None and opt.model != "")
            checkpoint = torch.load(opt.model, map_location=device)
            transformer.load_state_dict(checkpoint['model'])
        # First train
        train(transformer, training_data, validation_data, optimizer, device, opt, finetune=False)
        print('--- Before pruning ---')
        summary(transformer)
        exit()

    assert (opt.model != None and opt.model != "")
    
    print('--- Before pruning ---')
    summary(transformer)
    if opt.q != 0:
        # Prune
        prune(transformer, method='cascade', q=opt.q)
        print('--- After pruning ---')
        summary(transformer)
    # Finetune
    train(transformer, training_data, validation_data, optimizer, device, opt, finetune=True)
    summary(transformer)
    
def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train = TranslationDataset(
        fields=fields,
        path=opt.train_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()
