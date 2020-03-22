import warnings
class DefaultConfig(object):
    env='default'
    model='LSTMclassify'
    train_data_root=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/nlp-getting-started/train.csv'
    test_data_root=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/nlp-getting-started/test.csv'
    
    load_model_path=None
    #test:3263 train:7613
    batch_size=2000
    emb_size=128
    hidden_size=100
    n_layers=2
    model_attr=['emb_size','hidden_size','n_layers','batch_size']
    
    result_file='result.csv'
    
    max_epoch=20
    lr=0.01
    lr_decay=0.95
    #weight_decay=1e-2
    
    def parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('model has not attribute'+k)
            setattr(self,k,v)
        print('--------using config-----------')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self, k))