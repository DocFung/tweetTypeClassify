import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import re
import csv
import copy
import torch as t
#trainpath=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/nlp-getting-started/train.csv'
'''
class tweetsDisaster(data.Dataset):
    def __init__(self,filename,types='train'):
        self.keyword=[]
        self.location=[]
        self.text=[]
        self.target=[]
        with open(filename) as f:
            csvReader=csv.reader(f)
            #header
            next(csvReader)
            for row in csvReader:
                self.keyword.append(row[1])
                self.location.append(row[2])
                self.text.append(row[3])
                if types=='train':
                    self.target.append(row[4])
        if types !='train':
            self.target=['pred']*len(self.text)
        self.output=copy.copy(self.text)
    
    def dealwithText(self, dataset):
        wordDict=prepareWordDict(dataset)
        seq,lenth = data2seq(dataset, wordDict)
        self.output=list(zip(seq,lenth))

    def __getitem__(self, index):
        data=self.output[index]
        label=self.target[index]
        return data,label
    
    def __len__(self):
        return len(self.text)

'''
def loadData(path,types='train'):
    keyword=[]
    location=[]
    text=[]
    target=[]
    
    with open(path) as f:
        csvReader=csv.reader(f)
        header=next(csvReader)
        for row in csvReader:
            keyword.append(dealwithText(row[1]))
            location.append(dealwithText(row[2]))
            text.append(dealwithText(row[3]))
            target.append(row[4])
    
    #将target转换为1与-1的浮点数，方便计算loss
    for i,x in enumerate(target):
        if x=='0':
            target[i]=float(-1)
        else:
            target[i]=float(1)
    #target=t.autograd.Variable(t.Tensor(target))      
    return text,target

def dealwithText(text):
    text = re.sub('[^a-zA-Z]',' ',text)
    return text.strip()

def text2seq(dataset,word2index):
    dataSplit=[d.split() for d in dataset]
    sentenceLen=max([len(text) for text in dataSplit])
    datas=[]
    length=[]
    for text in dataSplit:
        vecText=[0]*sentenceLen
        vecText[:len(text)]=list(
            map(lambda w:word2index[w] if word2index[w] is not None else word2index['unk'], text))
        datas.append(vecText)
        length.append(len(text))
    return datas,length

def prepareWordDict(train):
    word2index={'unk':0}
    flatten = lambda l: [item for sublist in l for item in sublist]
    trainSplit=flatten([text.split() for text in train])
    for vo in trainSplit:
        if word2index.get(vo) is None:
            word2index[vo]=len(word2index)                   
    return word2index

