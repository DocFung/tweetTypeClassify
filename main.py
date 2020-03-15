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
#import fire
#import torchnet
opt=DefaultConfig()

def train(**kwargs):
    opt.parse(kwargs)
    #definition
    #这里想办法优化一下，尽量写成()
    model=getattr(models, opt.model)(opt.emb_size,opt.hidden_size,opt.n_layers,opt.batch_size)
    if opt.load_model_path:
        model.load(opt.load_model_path)
        
    #data
    text,target=D.loadData(opt.train_data_root)
    wordDict=D.prepareWordDict(text)
    seq,seqL=D.text2seq(text,wordDict)
    emb=t.nn.Embedding(len(wordDict),opt.emb_size)
    seqEmb=t.autograd.Variable(emb(t.LongTensor(seq)))
    target=t.autograd.Variable(t.Tensor(target))
    #loss and optimizer
    criterion=t.nn.MSELoss()
    lr=opt.lr
    optimizer=t.optim.Adam(model.parameters(),lr=lr,weight_decay=opt.weight_decay)
    
    #statistic index
    
    #train
    for epoch in range(opt.max_epoch):
        losses=[]
        hidden=model.init_hidden()
        for i,batch in enumerate(D.getBatch(seqEmb, seqL, target, opt.batch_size)) :
            inputs,inputsL,targets=batch[0],batch[1],batch[2]

            if len(inputs) != opt.batch_size:
                break
            model.zero_grad()
            output=model(inputs,inputsL,hidden)
        
            loss=criterion(output.view(-1), targets)
            losses.append(loss.item())
            loss.backward()
            t.nn.utils.clip_grad_norm(model.parameters(), 0.5) # gradient clipping
            optimizer.step()
            if i > 0 and i % 50 == 0:
                print("[%02d/%d] mean_loss : %0.2f, Perplexity : %0.2f" % (epoch,opt.max_epoch, np.mean(losses), np.exp(np.mean(losses))))
                losses = []
    opt.load_model_path=model.save()


def val():
    pass

def test(**kwargs):
    model = getattr(models, opt.model)(opt.emb_size,opt.hidden_size,opt.n_layers,opt.batch_size).eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    
    opt.parse(kwargs)
    text,target=D.loadData(opt.test_data_root,types='test')
    wordDict=D.prepareWordDict(text)
    seq,seqL=D.text2seq(text,wordDict)
    emb=t.nn.Embedding(len(wordDict),opt.emb_size)
    seqEmb=t.autograd.Variable(emb(t.LongTensor(seq)))
    #target=t.autograd.Variable(t.Tensor(target))
    pred=[]
    for i,batch in enumerate(D.getBatch(seqEmb, seqL, target, opt.batch_size)) :
            inputs,inputsL,targets=batch[0],batch[1],batch[2]
            hidden=model.init_hidden()
            if len(inputs) != opt.batch_size:
                break
            model.zero_grad()
            output=model(inputs,inputsL,hidden)
            pred.append(output.detach().contiguous().view(-1).numpy().tolist())
    flatten = lambda l: [item for sublist in l for item in sublist]
    pred=flatten(pred)
    
    for i,x in enumerate(pred):
        if x<0 or x==0:
            pred[i]=0
        else:
            pred[i]=1
    
    return pred

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
train()
pred=test()
name = time.strftime(opt.model+'_result'+ '%Y%m%d.csv')
name=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/result/'+name
write_csv(pred,name)