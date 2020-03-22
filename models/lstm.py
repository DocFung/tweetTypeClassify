# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:15:40 2020

@author: 46362
"""

import torch as t
import torch.nn as nn
from .basicModel import BasicModel
from torch import sigmoid
class LSTMclassify(BasicModel):
    
    def __init__(self,embSize,hiddenSize,nLayers,batchSize):
        super(LSTMclassify,self).__init__()
        self.modelName='LSTM'
        self.lstm=nn.LSTM(embSize,hiddenSize,nLayers,dropout=0.4)
        # in linear layer,output feature equal 2 mean two class classify using max
        self.linear=nn.Linear(hiddenSize,1)
        self.embSize=embSize
        self.nLayers=nLayers
        self.batchSize=batchSize
        self.hiddenSize=hiddenSize

    def init_weight(self):
        #self.embed.weight = nn.init.xavier_uniform(self.embed.weight)
        #self.linear.weight = nn.init.xavier_uniform(self.linear.weight,a=-1,b=1)
        nn.init.uniform_(self.linear.weight,a=-1,b=1)
        self.linear.bias.data.fill_(0)
        #self.embMatrix=[]
    def init_hidden(self):
        hidden = t.autograd.Variable(
            t.zeros(self.nLayers,self.batchSize,self.hiddenSize))
        return hidden
    def init_context(self):
        context = t.autograd.Variable(
            t.zeros(self.nLayers,self.batchSize,self.hiddenSize))
        return context


    
    def forward(self,inputs,inputsL,hidden,context):
        
        inputPack = t.nn.utils.rnn.pack_padded_sequence(inputs, inputsL, batch_first=True,enforce_sorted=False)
        outputPack, (hn,cn) = self.lstm(inputPack, (hidden,context))
        print('linear w')
        print('------------------------------')
        print(self.linear.weight)
    
        '''
        unpacked = t.nn.utils.rnn.pad_packed_sequence(outputPack,batch_first=True,total_length=33)
        #input one seq into rnn model, output a whole complete seq
        #use the final result of the seq, which refer to x[i-1]
        linearInput=[x[i-1] for x,i in zip(unpacked[0],unpacked[1])]
        linearInput=t.Tensor([x.detach().tolist() for x in linearInput])
        linearInput=linearInput.contiguous().view(linearInput.size(0),-1)
        '''
        print('hn------------------------')
        print(hn[-1])
        print('-------------------------------')
        print(sigmoid(self.linear(hn[-1])))
        return sigmoid(self.linear(hn[-1]))