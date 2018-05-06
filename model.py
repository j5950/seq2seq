# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:46:24 2018

@author: user
"""

import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
import numpy as np 

class Encoder(nn.Module):
    def __init__(self,voca_size, hidden_dim, embed_dim):
        super(Encoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(voca_size, embed_dim, padding_idx = 0)
        self.GRU = nn.GRU(embed_dim,hidden_dim, batch_first=True, bidirectional=True)
        self.weight_init()
        
    def weight_init(self):
        for weight in self.GRU.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal(weight.data) 
        #        nn.init.orthogonal(self.GRU.parameters().weight.data)
    
    def forward(self,input_seq, input_len):
        # input_seq : B x L
        embedded = self.embedding(input_seq) # return B x L x E (embed_dim)
        padded_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_len, batch_first=True)
        output, hid = self.GRU(padded_embedded) # B x L x 2 * H
        output,_ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hid
'''
enc = Encoder(10)
tmp = torch.LongTensor([[1,2,0],[4,5,0]])
tmp_var = Variable(tmp)
enc_h, hid = enc(tmp_var, [2, 2])
print((enc_h,hid))        
'''
class Attn(nn.Module):
    def __init__(self,hidden_dim):
        super(Attn,self).__init__()
        self.hidden_dim = hidden_dim
        self.W_a = nn.Linear(hidden_dim, hidden_dim) # dec_hid_dim x dec_hid_dim
        self.U_a = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        self.weight_init()

    def weight_init(self):
        nn.init.normal(self.W_a.weight,0,0.001)
        nn.init.normal(self.U_a.weight,0,0.001)
        nn.init.normal(self.linear.weight,0,0.01)
        
    def forward(self, prev_s, enc_h):
        # prev_s : B x H
        # enc_h : B x L x 2*H
        out_s = self.W_a(prev_s) # B x H
        out_s = out_s.unsqueeze(1) # B x 1 x H
        out_h = self.U_a(enc_h) # B x L x H
        expand_s = out_s.expand_as(out_h) # B x L x H
        out = F.tanh(expand_s + out_h)
        
        attn = F.softmax(self.linear(out)) # B x L x 1
        ctx = torch.bmm(enc_h.transpose(1,2), attn).transpose(1,2) # B x 1 x 2*H
        
        return ctx.squeeze(1) # B x 2*H
'''
attn = Attn(1000)
prev_s = hid[:,1,:]
ctx = attn(prev_s, enc_h)

print(ctx)
'''
class DecoderCell(nn.Module):
    def __init__(self,hidden_dim, embed_dim):
        super(DecoderCell,self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.W_emb_s = nn.Linear(embed_dim, hidden_dim) # s_tild
        self.W_emb_z = nn.Linear(embed_dim, hidden_dim) # update gate
        self.W_emb_r = nn.Linear(embed_dim, hidden_dim) # reset gate

        self.W_hid_s = nn.Linear(hidden_dim, hidden_dim) # s_tild
        self.W_hid_z = nn.Linear(hidden_dim, hidden_dim) # update gate
        self.W_hid_r = nn.Linear(hidden_dim, hidden_dim) # reset gate

        self.W_ctx_s = nn.Linear(2*hidden_dim, hidden_dim) # s_tild
        self.W_ctx_z = nn.Linear(2*hidden_dim, hidden_dim) # update gate
        self.W_ctx_r = nn.Linear(2*hidden_dim, hidden_dim) # reset gate
        self.weight_init()
    
    def weight_init(self):
        nn.init.normal(self.W_emb_s.weight,0,0.01)
        nn.init.normal(self.W_emb_z.weight,0,0.01)
        nn.init.normal(self.W_emb_r.weight,0,0.01)
        nn.init.normal(self.W_hid_s.weight,0,0.01)
        nn.init.normal(self.W_hid_z.weight,0,0.01)
        nn.init.normal(self.W_hid_r.weight,0,0.01)
        nn.init.normal(self.W_ctx_s.weight,0,0.01)
        nn.init.normal(self.W_ctx_z.weight,0,0.01)
        nn.init.normal(self.W_ctx_r.weight,0,0.01)
        
    def forward(self, prev_s, prev_emb, ctx):
        '''
        prev_s : B x H
        prev_emb : B x E
        ctx : B x 2*H
        ret : B x H
        '''
        z_i = F.sigmoid(self.W_emb_z(prev_emb) + self.W_hid_z(prev_s) + self.W_ctx_z(ctx))
        r_i = F.sigmoid(self.W_emb_r(prev_emb) + self.W_hid_r(prev_s) + self.W_ctx_r(ctx))
        s_tilde = F.tanh(self.W_emb_s(prev_emb) + self.W_hid_s(r_i * prev_s) + self.W_ctx_s(ctx))
        return (1-z_i) * prev_s + z_i * s_tilde
'''   
dec_cell = DecoderCell(1000)
dec_emb = torch.FloatTensor(2, 620).zero_()
dec_emb = Variable(dec_emb)
out = dec_cell(prev_s, dec_emb, ctx)
print(out)
'''
class Decoder(nn.Module):
    def __init__(self,hidden_dim, embed_dim, mx_len, voca_size, SOS_idx):
        super(Decoder,self).__init__()
        self.voca_size = voca_size
        self.mx_len = mx_len
        self.SOS_idx = SOS_idx
        
        self.embed = nn.Embedding(voca_size, embed_dim, padding_idx = 0)
        self.dec_cell = DecoderCell(hidden_dim, embed_dim)
        self.attn = Attn(hidden_dim)
        self.w_init = nn.Linear(hidden_dim,hidden_dim)
        self.hid2word = nn.Linear(hidden_dim, voca_size)
        
    def forward(self, enc_h, prev_s, target_word=None):
        '''
        enc_h : B x L x 2H
        target_wd == None <-> testing
        always teacher forcing --> arg받아서 처리하도 
        '''
        s_t = prev_s # hid state of h1 (inverse direction <--)
        ctx = None
        
        if target_word:
            batch_size, target_len = target_word.size(0), target_word.size(1)
            s_seq = Variable(torch.FloatTensor(batch_size,target_len, self.hidden_dim))
            target_emb = self.embed(target_word)

            for i in range(target_len):
                ctx = self.attn(s_t, enc_h)
                s_t = self.dec_cell(s_t, target_emb[:,i,:], ctx)
                s_seq[:,i,:] = s_t.unsqueeze(1)
                
            ret = self.hid2word(s_seq)
        else:
            batch_size = enc_h.size(0)
            target = Variable(torch.LongTensor(batch_size*[self.SOS_idx]))
            ret = Variable(torch.zeros(batch_size,self.mx_len, self.voca_size))
#            print(ret)
            for i in range(self.mx_len):
                target_emb = self.embed(target).squeeze(1)
                ctx = self.attn(s_t, enc_h)
                s_t = self.dec_cell(s_t,target_emb,ctx)
                ret[:,i,:] = self.hid2word(s_t)
                target = ret[:,i,:].topk(1)[1] # topk(k) --> top k elements[0], index[1] 
                
                
        return ret
'''        
dec = Decoder(1000)
ret = dec(enc_h, prev_s)

print(enc_h)
print(prev_s)
'''

class Seq2Seq(nn.Module):
    def __init__(self, hidden_dim, embed_dim, mx_len, src_voca_size, target_voca_size, SOS_idx):
        super(Seq2Seq,self).__init__()
        self.target_voca_size = target_voca_size
        self.encoder = Encoder(src_voca_size, hidden_dim, embed_dim)
        self.decoder = Decoder(hidden_dim, embed_dim, mx_len, target_voca_size, SOS_idx)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, src_seq, src_len, target=None):
        enc_h, enc_h_t = self.encoder(src_seq, src_len) # B x L x 2H, 2 x B x H
        # B x H
        enc_h_t = enc_h_t[1] # B x H
        s_0 = F.tanh(self.linear(enc_h_t))
        
        ret = self.decoder(enc_h, s_0, target)
        ret = F.log_softmax(ret.contiguous().view(-1,self.target_voca_size))
        
        return ret