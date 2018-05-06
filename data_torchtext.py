import os 

import torch
from torchtext import data,datasets 
from torchtext.vocab import Vectors
import re 
import spacy
import numpy as np 

from spacy.language import Language
from torchtext.vocab import FastText
import pdb 

def tokenize_lang(text,tokenizer):
   url = re.compile('(<url>.*</url>)')
   return [tok.text for tok in tokenizer(url.sub('@URL@', text))]

def get_data(data_name,multi_language=False,embedding_size=300,src='.de',trg='.en',tokenizer_name='moses',min_count=5,max_vocab_size=50000,use_vocab = True,sos_token = '<s>',eos_token = '</s>',lower=True,batch_first =False):

    
   """
      Get dataset funciton
      
      Args:
         
         data_name
         embedding_size 
         src
         tgt
         tokenizer_name 
         max_vocab_size 
         use_vocab
         sos_token
         eos_token 
         lower 
         batch_first

      returns:
         
         train
         val
         test
         src_vocab
         tgt_vocab

   """

    # Torch Text Preprocessing Data Filed Define

   tokenizer = data.get_tokenizer(tokenizer_name) # $$ need to modify this part -> make more flexible  
   
   src_lang_text = data.Field(tokenize = tokenizer,init_token=sos_token,eos_token=eos_token,\
                        use_vocab = use_vocab,lower=lower,batch_first=batch_first)   
   trg_lang_text = data.Field(tokenize = tokenizer,init_token=sos_token,eos_token=eos_token,\
                        use_vocab = use_vocab,lower=lower,batch_first=batch_first)
   torch_text_preload = ['WMT14DE','IWSLT','multi30k']
   print ("[+] Get the data")
   # torchtext library supported only de - english dataset. 

   if data_name in torch_text_preload:
      if src == '.de':
         if data_name == 'WMT14DE':
            train,val,test = datasets.WMT14.splits(exts = ('.de','.en'),fields = (src_lang_text,trg_lang_text)) 
         elif data_name == 'IWSLT':
            train,val,test = datasets.IWSLT.splits(exts = (src,trg),fields = (src_lang_text,trg_lang_text))
         else: 
            train,val,test = datasets.Multi30k.splits(exts =(src,trg),fields = (src_lang_text,trg_lang_text) )
      else:
         # Source other languages 
         raise NotImplementedError
   else:
      pdb.set_trace()
      train,val,test = datasets.TranslationDataset.splits(path=data_name,train='train',validation='val',
                                             exts =(src,trg),fields = (src_lang_text,trg_lang_text) )
      raise NotImplementedError
   
   # Drop Max_lengths:
   # build vocab
   src_lang_text.build_vocab(train.src)
   trg_lang_text.build_vocab(train.trg)

   src_vocab = src_lang_text.vocab
   trg_vocab = trg_lang_text.vocab
   print ("[+] Vocabulary size of src language : {} / tgt_language : {}".format(len(src_vocab),len(trg_vocab)))
   print ("[+] Trim the data most_frequent {} words & discard words appear than {} times ".format(max_vocab_size,min_count))

   #$$ need to make flexible word vector inputs
   print ("[+] Load Pretrained Word embedding ")
   vector_cache = os.path.join('/tmp', 'vector_cache')
   if multi_language: 
      trg_lang_text.build_vocab(train.trg,min_freq=min_count,max_size = max_vocab_size, 
                        vectors=Vectors('wiki.multi.en.vec',cache = vector_cache,url='https://s3.amazonaws.com/arrival/embeddings/wiki.multi.en.vec'))
      src_lang_text.build_vocab(train.src,min_freq=min_count,max_size = max_vocab_size, 
                        vectors = Vectors('wiki.multi.de.vec',cache = vector_cache,url='https://s3.amazonaws.com/arrival/embeddings/wiki.multi.de.vec'))
   else:
      trg_lang_text.build_vocab(train.trg,min_freq=min_count,max_size = max_vocab_size, vectors =  FastText(language=trg[1:],cache = vector_cache))
      src_lang_text.build_vocab(train.src,min_freq=min_count,max_size = max_vocab_size, vectors =  FastText(language=src[1:],cache = vector_cache))

   # Initialize <unk>,<s>,</s>
   
   emb_dims = src_lang_text.vocab.vectors.size()[1]
   std = 1.0/np.sqrt(emb_dims)
   if multi_language:
      value = np.random.normal(0,scale =std ,size=[2,emb_dims]) 
      value = torch.from_numpy(value)
      src_lang_text.vocab.vectors[1:3] = value
      trg_lang_text.vocab.vectors[1:3] = value
   else:
      value1 = np.random.normal(0,scale =std ,size=[3,emb_dims])          
      value2 = np.random.normal(0,scale =std ,size=[3,emb_dims])
      value1 = torch.from_numpy(value1)
      value2 = torch.from_numpy(value2)
   
      src_lang_text.vocab.vectors[1:4] = value1
      trg_lang_text.vocab.vectors[1:4] = value2

   src_vocab = src_lang_text.vocab
   trg_vocab = trg_lang_text.vocab
   # need  to check word embedding 
   
   print ("[+] End of the load dataset")
   
   """
   print ("[+] Load custom Word vector")
   nlp = Language()
   with open('/hdd1/muse_pretrained/wiki.multi.fr.vec.l') as file_:
      header = file_.readline()
      nr_row,nr_dim = header.split()
      nlp.vocab.reset_vectors(width = int(nr_dim))
      for line in file_:
         line =  line.rstrip().decode('utf8')
         pieces = line.rstrip(' ',int(nr_dim))
         word = pieces[0]
         vector = numpy.array([float (v) for v in pieces[1:]],dtype = 'f')
         nlp.vocab.set_vector(word,vector)
   """
   return train,val,test,src_vocab,trg_vocab

