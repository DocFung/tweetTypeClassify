testpath=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/nlp-getting-started/train.csv'
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
import model
#model1= getattr('model', 'simple_rnn')()
#model1=model.RNNclassiy(vocSize, embSize, hiddenSize, nLayers, batchSize)
text,target=D.loadData(testpath)
wordDict=D.prepareWordDict(text)
seq,seqL=D.text2seq(text,wordDict)

emb=t.nn.Embedding(len(wordDict),128)
seqEmb=t.autograd.Variable(emb(t.LongTensor(seq)))