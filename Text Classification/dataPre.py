# -*- coding: utf-8 -*-

import torch
import re
import pandas as pd
from itertools import chain
import torch.nn as nn


class Data(object):

    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, fdata):
        self.fdata = fdata
        self.train_p,self.val_p = self.dataLoad(fdata)
        self.vocab,_,_ = self.preprocess()
        self.vocab = [self.PAD, self.UNK] + self.vocab
        self.vocab_size = len(self.vocab)
        self.word_to_idx  = {word: i for i, word in enumerate(self.vocab)}
        self.unk_idx = self.word_to_idx[self.UNK]
        self.pad_idx = self.word_to_idx[self.PAD]
        
    def produce_data(self,maxlen):
        print('Start generate data...')
        _,train_tokenized,val_tokenized = self.preprocess()
        print('训练样本数',len(train_tokenized))
        print('测试样本数',len(val_tokenized))
        train_x = self.pad_samples(self.encode_samples(train_tokenized), maxlen)
        val_x = self.pad_samples(self.encode_samples(val_tokenized), maxlen)
        train_y = [tar for tar in self.train_p['sentiment']]
        val_y = [tar for tar in self.val_p['sentiment']]
        print('End generate data...')
        
        return train_x,train_y,val_x,val_y
        
        
    def encode_samples(self,tokenized_samples):#分词之后的文本，如果在就添加index，不在就记为0
        features = []
        for sample in tokenized_samples:
            feature = []
            for token in sample:
                if token in self.word_to_idx:
                    feature.append(self.word_to_idx[token])
                else:
                    feature.append(self.unk_idx)
            features.append(feature)
                
        return features

    def pad_samples(self,features, maxlen):#长于设定序列长度就切掉，不足就补0
        padded_features = []
        for feature in features:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
                while(len(padded_feature) < maxlen):
                    padded_feature.append(self.pad_idx)
            padded_features.append(padded_feature)
            
        return padded_features
    
    def extend(self,words):
        unk_words = [w for w in words if w not in self.word_to_idx]
        self.vocab = sorted(set(self.vocab + unk_words) - {self.PAD})
        self.vocab = [self.PAD] + self.vocab
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.pad_idx = self.word_to_idx['<PAD>']
        self.unk_idx = self.word_to_idx['<UNK>']
        self.vocab_size = len(self.vocab)
    
    def init_embedding(self, tensor):
        std = (1. / tensor.size(1)) ** 0.5
        nn.init.normal_(tensor, mean=0, std=std)
        
    def get_embed(self,fembed):
        print('Start generate embed...')
        with open(fembed, 'r',encoding='UTF-8') as f:
            lines = [line for line in f]
        splits = [line.split(' ') for line in lines]
        words = [split[0] for split in splits]
        embed = [list(map(float,split[1:])) for split in splits]
        self.extend(words)    
        embed = torch.tensor(embed, dtype=torch.float)
        embed_indices = [self.word_to_idx[w] for w in words]
        extended_embed = torch.Tensor(self.vocab_size, embed.size(1))
        self.init_embedding(extended_embed)
        extended_embed[embed_indices] = embed
        print('End generate embed...')

        return extended_embed
    
    def dataLoad(self,fdata):
        data = pd.read_csv(fdata, header=0, delimiter="\t", quoting=3)
        train_p = data.iloc[0:23000]
        val_p = data.iloc[23000:]
        
        return train_p,val_p
    
    def review_to_wordlist(self,review):
        review_text=re.sub('\'ve',' have',review)
        review_text=re.sub('n\'t',' not',review_text)
        review_text=re.sub('<br />','',review_text)
        review_text = re.sub("[^a-zA-Z]",' ', review_text)
        words = review_text.lower().split()
    
        return words
        
    def preprocess(self):
        train_p,val_p = self.dataLoad(self.fdata)
        train_tokenized = []
        val_tokenized = []
        for review in train_p['review']:
            train_tokenized.append(self.review_to_wordlist(review))
        for review in val_p['review']:
            val_tokenized.append(self.review_to_wordlist(review))  
        vocab = set(chain(*(train_tokenized + val_tokenized)))
        
        return list(vocab),train_tokenized,val_tokenized

        
        
        
        



    

