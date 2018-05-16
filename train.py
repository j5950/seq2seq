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
num_epoch = 10
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



def train(model, src_seqs, src_lens, trg_seqs, optimizer, criterion, is_valid):
    batch_size = src_seqs.size(0)


    src_seqs_var = Variable(src_seqs.data.clone())
    trg_seqs_var = Variable(trg_seqs[:, :-1].data.clone()) # from <sos> to before <eos>

    out = model(src_seqs_var, src_lens, trg_seqs_var) # L x V
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



def save_results(model, src_seqs, src_lens,trg_seqs):
    print("save results sentence!")
    src_seqs_var = Variable(src_seqs.data.clone())
    labels = trg_seqs.clone() # B x L

    out = model(src_seqs_var, src_lens)
    pred = out.max(2)[1]
    file_name = "results_ep"+str(num_epoch)+".txt"
    f = open(file_name, 'a')

    for i in range(labels.size(0)):
        answer = ""
        output = ""
        for wd in labels[i].data:
            word = trg_vocab.itos[wd]
            answer = answer + ' ' + word
            if word == '</s>':
                answer += '\n'
                break
        for wd in pred[i].data:
            word = trg_vocab.itos[wd]
            output = output + ' ' + word
            if word == '</s>':
                output += '\n'
                break

        f.write('\n')
        f.write('answer : ' + answer)
        f.write('output : ' + output)
    f.close()


seq2seq = Seq2Seq(hidden_dim,embed_dim,mx_len,src_vocab,trg_vocab,trg_vocab.stoi['<s>'], dropout_p, num_layer_gru)

if cuda.is_available():
    seq2seq.cuda()

optim_parameters = filter(lambda p: p.requires_grad, seq2seq.parameters())
optimizer = optim.Adadelta(optim_parameters, rho=0.95)
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi['<pad>'])

for ep in range(num_epoch):
    cnt = 0
    sum_val_loss = 0
    for val_batch in valid_iter:
        sum_val_loss = sum_val_loss + train(seq2seq, val_batch.src[0], val_batch.src[1].tolist(), val_batch.trg[0],
                                            optimizer, criterion, is_valid=True)
        cnt = cnt + 1
    print("\nEpoch #{0} : val avg loss = {1}\n".format(ep, sum_val_loss/cnt))

    for (i, train_batch) in tqdm(enumerate(train_iter)):

        src_seq = train_batch.src[0]
        src_len = train_batch.src[1]
        trg_seq = train_batch.trg[0]
        loss, num_correct, num_total = train(seq2seq, src_seq, src_len.tolist(), trg_seq,
                                             optimizer, criterion, is_valid=False)


        if i % 100 == 0:
            print("\n{0} Epoch : Train loss is {1}\n".format(ep, loss))
            print("correct / total =")
            print(num_correct)
            print("/")
            print(num_total)


for (i, test_batch) in tqdm(enumerate(test_iter)):
    src_seq = test_batch.src[0]
    src_len = test_batch.src[1]
    trg_seq = test_batch.trg[0]
    trg_len = test_batch.trg[1]
    save_results(seq2seq, src_seq, src_len.tolist(), trg_seq)

