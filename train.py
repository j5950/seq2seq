# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:18:16 2018

@author: Junha
"""

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from tensorboardX import SummaryWriter
import numpy as np

import pdb
from tqdm import tqdm

from model import *
from data_torchtext import *

#pdb.set_trace()
#=============================== Hyperparams ==========================================#
hidden_dim = 1000
mx_len = 50
batch_size = 32
num_iter = 5
embed_dim = 300
dropout_p = 0.5
num_epoch = 8
num_layer_gru = 2
tensorboard_dir = './tensorboard'
#=============================== Load data file =======================================#
train_data, val_data, test_data, src_vocab, trg_vocab = get_data(
    data_name='multi30k', batch_first=True)
train_iter = data.BucketIterator(
    dataset=train_data, repeat=False, shuffle=True, batch_size=batch_size, device=0, sort_within_batch=True, sort_key=lambda x: len(x.src))
valid_iter = data.BucketIterator(
    dataset=val_data, repeat=False, shuffle=True, batch_size=batch_size, device=0, sort_within_batch=True, sort_key=lambda x: len(x.src))
test_iter = data.BucketIterator(
    dataset=test_data, repeat=False, shuffle=True, batch_size=batch_size, device=0, sort_within_batch=True, sort_key=lambda x: len(x.src))


'''for train_batch in train_iter:
    print(train_batch.src[1][1])
    print(train_batch.trg[1][1])
    break'''



def train(model, src_seqs, src_lens, trg_seqs, optimizer, criterion, is_valid):
    batch_size = src_seqs.size(0)

# cuda, dev는 constant로 받기

    src_seqs_var = Variable(src_seqs.data.clone())
    trg_seqs_var = Variable(trg_seqs[:, :-1].data.clone()) # from <sos> to before <eos>

    out = model(src_seqs_var, src_lens, trg_seqs_var) # L x V
    '''    tmp = out.max(2)[1]
    print("out size : {0}\n".format(out.size()))
    print("model outcome")
    print(tmp[0])
    print("answer")
    print(trg_seqs[0])'''
    out = out.view(-1, len(trg_vocab)).contiguous() # B x L x V -> BL x V

    labels = trg_seqs[:, 1:].contiguous().view(-1) # except <sos>

    loss = 0

    if is_valid:
        loss = criterion(out, labels)
        return loss

    optimizer.zero_grad()
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    pred = out.max(1)[1]
    num_corrects = pred.data.eq(labels.data)
    num_corrects = num_corrects.masked_select(labels.data.ne(trg_vocab.stoi['<pad>'])).sum() # pad idx == 0 ?
    num_total = labels.data.ne(trg_vocab.stoi['<pad>']).sum()
    return loss, num_corrects, num_total


def eval(model, src_seqs, src_lens, trg_seqs, trg_lens):
    trg_seqs = trg_seqs[:, 1:]
    src_seqs_var = Variable(src_seqs.data.clone())

    out = model(src_seqs_var, src_lens)
    labels = trg_seqs.clone() # B x L
#    trg_lens_np = trg_lens.cpu().numpy()
#    labels = labels[np.arange(0, batch_size), trg_lens_np]
#    labels = trg_seqs.contiguous().view(-1)

#    out = out.view(-1, trg_voca_size).contiguous() # B x L x V -> BL x V

    pred = out.max(2)[1]
#    print("@@")
#    print(pred.size(-1))
#    print(labels.size(-1))
  #  num_corrects = pred.data.eq(labels.data)
  #  num_corrects = num_corrects.masked_select(labels.data.ne(0)).sum() # pad idx == 0 ?

    for i in range(batch_size):
        print("answer")
        print(labels[i].data)
        print("length")
        print(trg_lens[i])
        print("predict")
        print(pred[i])


def save_results(model, src_seqs, src_lens,trg_seqs):
    print("save results sentence!")
    src_seqs_var = Variable(src_seqs.data.clone())
    labels = trg_seqs.clone() # B x L

    out = model(src_seqs_var, src_lens)
    pred = out.max(2)[1]

    ref_f = open("reference.txt", 'a')
    pred_f = open("predictions.txt", 'a')
    for i in range(labels.size(0)):
        answer = ""
        output = ""
        for wd in labels[i].data:
            word = trg_vocab.itos[wd]
            if word == '<s>':
                continue
            if word == '</s>':
                answer = answer.rstrip() + '\n'
                break
            answer = answer + word + ' '
        for wd in pred[i].data:
            word = trg_vocab.itos[wd]
            if word == '</s>':
                output = output.rstrip() + '\n'
                break
            output = output + word + ' '

        ref_f.write(answer)
        pred_f.write(output)
    ref_f.close()
    pred_f.close()


seq2seq = Seq2Seq(hidden_dim,embed_dim,mx_len,src_vocab,trg_vocab,trg_vocab.stoi['<s>'], dropout_p, num_layer_gru)

if cuda.is_available():
    seq2seq.cuda()

optim_parameters = filter(lambda p: p.requires_grad, seq2seq.parameters())
optimizer = optim.Adadelta(optim_parameters, rho=0.95)
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi['<pad>'])

for ep in range(num_epoch):

    for (i, train_batch) in tqdm(enumerate(train_iter)):

        src_seq = train_batch.src[0]
        src_len = train_batch.src[1]
        trg_seq = train_batch.trg[0]
    #    print(src_len)
    #    print(src_seq)
        loss, num_correct, num_total = train(seq2seq, src_seq, src_len.tolist(), trg_seq,
                                             optimizer, criterion, is_valid=False)


        if i % 100 == 0:
            print("\n{0} Epoch : Train loss is {1}\n".format(ep, loss))
            print("correct / total =")
            print(num_correct)
            print("/")
            print(num_total)

    cnt = 0
    sum_val_loss = 0
    for val_batch in valid_iter:
        sum_val_loss = sum_val_loss + train(seq2seq, val_batch.src[0], val_batch.src[1].tolist(), val_batch.trg[0],
                                            optimizer, criterion, is_valid=True)
        cnt = cnt + 1
    print("\nEpoch #{0} : val avg loss = {1}\n".format(ep+1, sum_val_loss/cnt))

for (i, test_batch) in tqdm(enumerate(test_iter)):
    src_seq = test_batch.src[0]
    src_len = test_batch.src[1]
    trg_seq = test_batch.trg[0]
    trg_len = test_batch.trg[1]
    save_results(seq2seq, src_seq, src_len.tolist(), trg_seq)

