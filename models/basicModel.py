import torch as t
import torch.nn as nn
import time

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel,self).__init__()
        self.modelName=str(type(self))

    def load(self,path):
        self.load_state_dict(t.load(path))
    
    def save(self,name=None):
        if name is None:
            prefix='checkpoint/'+self.modelName+'_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(),name)
        return name
        