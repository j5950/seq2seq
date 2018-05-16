# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:46:24 2018

@author: user
"""
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, src_voca, hidden_dim, embed_dim, dropout_p, num_layer_gru):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(len(src_voca), embed_dim, padding_idx=1)
        self.embed.weight.data.copy_(src_voca.vectors)
        self.embed.weight.requires_grad = False
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout_p, num_layers=num_layer_gru)
        self.weight_init()
        
    def weight_init(self):
        for weight in self.gru.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal(weight.data) 


    def forward(self, input_seq, input_len):
        '''
        :param input_seq: B x L
        :param input_len: B (Not Variable)
        :return: output: B x L x H (PackedSequence)
        '''

        embedded = self.embed(input_seq) # return B x L x E (embed_dim)
        padded_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_len, batch_first=True)
        output, hid = self.gru(padded_embedded) # B x L x 2 * H
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hid


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
        
        attn = F.softmax(self.linear(out), dim=1) # B x L x 1
        ctx = torch.bmm(enc_h.transpose(1, 2), attn).transpose(1, 2) # B x 1 x 2*H
        
        return ctx.squeeze(1) # B x 2*H


class DecoderCell(nn.Module):
    def __init__(self, hidden_dim, embed_dim):
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


class Decoder(nn.Module):
    def __init__(self,hidden_dim, embed_dim, mx_len, trg_voca, SOS_idx):
        super(Decoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.voca_size = len(trg_voca)
        self.mx_len = mx_len
        self.SOS_idx = SOS_idx
        
        self.embed = nn.Embedding(self.voca_size, embed_dim, padding_idx=1)
        self.embed.weight.data.copy_(trg_voca.vectors)
        self.embed.weight.requires_grad = False

        self.dec_cell = DecoderCell(hidden_dim, embed_dim)
        self.attn = Attn(hidden_dim)
        self.w_init = nn.Linear(hidden_dim,hidden_dim)
        self.hid2word = nn.Linear(hidden_dim, self.voca_size)
        
    def forward(self, enc_h, prev_s, target_word=None):
        '''
        enc_h : B x L x 2H
        target_wd == None <-> testing
        always teacher forcing --> arg받아서 처리하도 
        '''

#        if cuda.is_available():
#            self.dec_cell.cuda()
 #           self.attn.cuda()
#            self.hid2word.cuda()
        s_t = prev_s # hid state of h1 (inverse direction <--)
        ctx = None
        
        if target_word is not None:
            batch_size, target_len = target_word.size(0), target_word.size(1)
            s_seq = Variable(torch.FloatTensor(batch_size,target_len, self.hidden_dim))
            if cuda.is_available():
           #     print("s_seq -> cuda!!")
                s_seq = s_seq.cuda()
            #    print(s_seq)
            target_emb = self.embed(target_word)

            for i in range(target_len):
                ctx = self.attn(s_t, enc_h)
                s_t = self.dec_cell(s_t, target_emb[:,i,:], ctx)
                s_seq[:, i, :] = s_t.unsqueeze(1)
                
            ret = self.hid2word(s_seq) # B x L x V
        else:
            batch_size = enc_h.size(0)
            target = Variable(torch.LongTensor(batch_size*[self.SOS_idx]))
            ret = Variable(torch.zeros(batch_size, self.mx_len, self.voca_size))

            if cuda.is_available():
                target = target.cuda()
                ret = ret.cuda()
#            print(ret)

            for i in range(self.mx_len):
                target_emb = self.embed(target).squeeze(1)
                ctx = self.attn(s_t, enc_h)
                s_t = self.dec_cell(s_t,target_emb,ctx)
                ret[:, i, :] = self.hid2word(s_t)
                target = ret[:, i, :].topk(1)[1] # topk(k) --> top k elements[0], index[1]

        return ret # B x L x V


class Seq2Seq(nn.Module):
    def __init__(self, hidden_dim, embed_dim, mx_len, src_voca, target_voca, SOS_idx, dropout_p, num_layer_gru):
        super(Seq2Seq, self).__init__()
        self.target_voca_size = len(target_voca)
        self.encoder = Encoder(src_voca, hidden_dim, embed_dim, dropout_p, num_layer_gru)
        self.decoder = Decoder(hidden_dim, embed_dim, mx_len, target_voca, SOS_idx)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, src_seq, src_len, target=None):

        enc_h, enc_h_t = self.encoder(src_seq, src_len) # B x L x 2H, 2 x B x H
        # B x H
        enc_h_t = enc_h_t[1] # B x H
        s_0 = F.tanh(self.linear(enc_h_t))
        
        ret = self.decoder(enc_h, s_0, target)
#        ret = F.log_softmax(ret.contiguous().view(-1, self.target_voca_size), dim=-1)# BL x V
        ret = F.log_softmax(ret, dim=-1) # B x L x V
        return ret
