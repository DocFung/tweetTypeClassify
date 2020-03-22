#testpath=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/nlp-getting-started/train.csv'
'''
from data.dataset import tweetsDisaster
from torch.utils.data import DataLoader
test=tweetsDisaster(testpath)
test.dealwithText(test.text)

testLoader=DataLoader(test)
testiter=iter(testLoader)
print(next(testiter))
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
#import fire
#import torchnet
opt=DefaultConfig()

path=get_tmpfile("word2vec.model")
def train(**kwargs):
    opt.parse(kwargs)
    RESCHEDULED = False
    #definition
    #这里想办法优化一下，尽量写成()
    model=getattr(models, opt.model)(opt.emb_size,opt.hidden_size,opt.n_layers,opt.batch_size)
    model.init_weight() 
    if opt.load_model_path:
        model.load(opt.load_model_path)
        
    #data
    #生成词字典，将句子转换为向量
    trainText,target=D.loadData(opt.train_data_root)
    wordDict=D.prepareWordDict(trainText)
    seq,seqL=D.text2seq(trainText,wordDict)
    #根据word2vec模型生成新字典，将权重导入embedding
    trainText=[x.split() for x in trainText]
    #这里发现word2vec得到的词向量会过小导致全连接层后以及sigmoid后都偏于0
    word2VecModel=Word2Vec(trainText,size=opt.emb_size,window=5,min_count=1,workers=4)
    emb=t.nn.Embedding.from_pretrained(t.FloatTensor(word2VecModel.wv.vectors))
    #emb=t.nn.Embedding(len(wordDict),opt.emb_size)
    #emb.weight = t.nn.init.xavier_uniform(emb.weight)
    t.nn.init.normal_(emb.weight,mean=0,std=1)
    seqEmb=t.autograd.Variable(emb(t.LongTensor(seq)))
    target=t.autograd.Variable(t.Tensor(target))

    #loss and optimizer
    #criterion=t.nn.MSELoss()
    #criterion2=t.nn.functional.binary_cross_entropy()
    
    lr=opt.lr
    #optimizer=t.optim.Adam(model.parameters(),lr=lr,weight_decay=opt.weight_decay)
    optimizer=t.optim.Adam(model.parameters(),lr=lr)
    #optimizer=t.optim.SGD(model.parameters(),lr=lr)
    #statistic index
    
    #train
    for epoch in range(opt.max_epoch):
        losses=[]
        hidden=model.init_hidden()
        context=model.init_context()
        for i,batch in enumerate(D.getBatch(seqEmb, seqL, target, opt.batch_size)):
            
            inputs,inputsL,targets=batch[0],batch[1],batch[2]

            if len(inputs) != opt.batch_size:
                break
            model.zero_grad()
            output=model(inputs,inputsL,hidden,context)
            output=output.contiguous().view(-1)

            #output=output.detach().view(2,-1)
            targets=targets.float()#the crossentropy require
            loss=t.nn.functional.binary_cross_entropy(output,targets)
            losses.append(loss.item())
            loss.backward()
            #t.nn.utils.clip_grad_norm(model.parameters(), 0.5) # gradient clipping
            optimizer.step()
            #if i > 0 and i % 50 == 0:
            print("[%02d/%d] loss : %0.2f" % (epoch,opt.max_epoch, np.mean(losses)))
            losses=[]
        if RESCHEDULED == False and epoch == opt.max_epoch//2:
            optimizer = t.optim.Adam(model.parameters(), lr=lr*0.1)
            RESCHEDULED = True
    opt.load_model_path=model.save()

    

def val():
    pass

def test(**kwargs):
    opt.parse(kwargs)
    model = getattr(models, opt.model)(opt.emb_size,opt.hidden_size,opt.n_layers,opt.batch_size).eval()
    if opt.load_model_path:
        print('loading------------------------')
        model.load(opt.load_model_path)
    
    #读数据
    text,target=D.loadData(opt.test_data_root,types='test')
    wordDict=D.prepareWordDict(text)
    seq,seqL=D.text2seq(text,wordDict)
    #生成词向量
    text=[x.split() for x in text]
    word2VecModel=Word2Vec(text,size=opt.emb_size,window=5,min_count=1,workers=4)
    emb=t.nn.Embedding.from_pretrained(t.FloatTensor(word2VecModel.wv.vectors))
    seqEmb=t.autograd.Variable(emb(t.LongTensor(seq)))
    
    #target=t.autograd.Variable(t.Tensor(target))
    
    for i,batch in enumerate(D.getBatch(seqEmb, seqL, target, opt.batch_size)) :
            inputs,inputsL,targets=batch[0],batch[1],batch[2]
            hidden=model.init_hidden()
            context=model.init_context()
            #偷懒处理
            if len(inputs) != opt.batch_size:
                break
            model.zero_grad()

            output=model(inputs,inputsL,hidden,context)
            #pred.append(output)
            #pred.append(output.detach().contiguous().view(-1).numpy().tolist())
    #flatten = lambda l: [item for sublist in l for item in sublist]
    #pred=flatten(pred)
    
    pred=[]
    '''
    for x in output:
        if x[0]>x[1]:
            pred.append(0)
        else:
            pred.append(1)
    '''
    for i,x in enumerate(output):
        if x<0.5:
            pred.append(0)
        else:
            pred.append(1)
    
    return pred,sum(pred)

def write_csv(pred,filename):
    with open(filename,'w') as f:
        writer=csv.writer(f)
        writer.writerow(pred)

def help():
    pass
'''
if __name__=='__main__':
    import fire
    fire.Fire()

#model1=models.RNNclassiy(1, 1, 1, 1, 1)
text,target=D.loadData(testpath)
wordDict=D.prepareWordDict(text)
seq,seqL=D.text2seq(text,wordDict)

emb=t.nn.Embedding(len(wordDict),opt.emb_size)
seqEmb=t.autograd.Variable(emb(t.LongTensor(seq)))

model1= getattr(models, 'RNNclassiy')(opt.emb_size,opt.hidden_size,opt.n_layers,opt.batch_size)
'''
train(batch_size=7613)
res,res_sum=test(batch_size=3263)
#name = time.strftime(opt.model+'_result'+ '%Y%m%d%H%M.csv')
#name=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/result/'+name
#write_csv(pred,name)
