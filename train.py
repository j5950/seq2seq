# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:18:16 2018

@author: Junha
"""

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import math
import time
import os
import pdb

from model import *
from data_torchtext import *

#pdb.set_trace()
train_data, val_data, test_data, src_vocab, trg_vocab = get_data(data_name='multi30k',embedding_size=620,batch_first=True)
print("@@")
#=============================== Hyperparams ==========================================#
"""hidden_dim=
src_voca_size=
trg_voca_size=
mx_len=
learning_rate=
batch_size=
num_iter=
"""

#=============================== Load data file =======================================#

#============================== Define Model ==========================================#
#my_model = Seq2Seq(hidden_dim)
#======================================================================================#