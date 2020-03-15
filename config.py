import warnings
class DefaultConfig(object):
    env='default'
    model='RNNclassify'
    train_data_root=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/nlp-getting-started/train.csv'
    test_data_root=r'C:/Users/46362/Desktop/home/AI/kaggle datasets/Real or Not NLP with Disaster Tweets/nlp-getting-started/test.csv'
    
    load_model_path=None
    
    batch_size=5
    emb_size=128
    hidden_size=20
    n_layers=2
    model_attr=['emb_size','hidden_size','n_layers','batch_size']
    
    result_file='result.csv'
    
    max_epoch=10
    lr=0.1
    lr_decay=0.95
    weight_decay=1e-4
    
    def parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('model has not attribute'+k)
            setattr(self,k,v)
        print('--------using config-----------')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self, k))