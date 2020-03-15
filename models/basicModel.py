import torch as t
import torch.nn as nn
import time
#path=C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/model/
class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel,self).__init__()
        self.modelName=str(type(self))

    def load(self,path):
        self.load_state_dict(t.load(path))
    
    def save(self,name=None):
        if name is None:
            prefix='checkpoint/'+self.modelName+'_'
            name = time.strftime(prefix + '%Y%m%d.pth')
        t.save(self.state_dict(),name)
        return name
        