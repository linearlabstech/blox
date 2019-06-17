import torch
from torch import nn
import sklearn

class SKL(nn.Module):

    '''
        A simple wrapper around the sklearn library to use in downstream tasks
    '''

    def __init__(self,mtype,**kwargs):
        model = sklearn
        for m in mtype.split('.'):model = getattr(model,m)
        self.model = model(**kwargs)

    def forward(self,x):
        if isinstance(x,torch.Tensor):
            if torch.cuda.is_available():x = x.cpu()
            return torch.from_numpy(self.model.predict(x.data.numpy()))
        return torch.from_numpy(self.model.predict(x))
    
    def load_state_dict(self,fname):pass