# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:50:58 2020

@author: 46362
"""
'''
import torch as t
import data.dataset as D
import models
from config import DefaultConfig
import numpy as np
import csv
import time
from gensim.test.utils import common_texts,get_tmpfile
from gensim.models import Word2Vec
opt=DefaultConfig()
#生成词字典，将句子转换为向量
trainText,target=D.loadData(opt.train_data_root)
wordDict=D.prepareWordDict(trainText)
seq,seqL=D.text2seq(trainText,wordDict)
#根据word2vec模型生成新字典，将权重导入embedding
trainText=[x.split() for x in trainText]
word2VecModel=Word2Vec(trainText,size=opt.emb_size,window=5,min_count=1,workers=4)
emb=t.nn.Embedding.from_pretrained(t.FloatTensor(word2VecModel.wv.vectors))

#word2VecModel=Word2Vec(trainText,size=opt.emb_size,window=5,min_count=1,workers=4)
#emb=t.nn.Embedding.from_pretrained(t.FloatTensor(word2VecModel.wv.vectors))
#emb=t.nn.Embedding(len(trainWordDict),128)
emb_seq=t.autograd.Variable(emb(t.LongTensor(seq)))
'''
# input is of size N x C = 3 x 5
import torch
import torch.nn.functional as F
input = torch.sigmoid(torch.randn(5, requires_grad=True))
# each element in target has to have 0 <= value < C
targets = torch.tensor([1,0,0,1,1]).float()
print('torch-----------------------------')
outputs = F.binary_cross_entropy(input, targets)
print(outputs)
def MyCrossEntropy(input,target):
    score=torch.tensor(0).float()
    for x,y in zip(input,target):
        score+= y*x.log()+(1-y)*(1-x).log()
    return -1*score/5

print('mine------------------------------')
print(MyCrossEntropy(input, targets))
def pred2res(L):
    res=[]
    for i in L:
        if i[0]>i[1]:
            res.append(0)
        else:
            res.append(1)
    return res
