

import torch
# from torch.nn.modules.loss import _WeightedLoss
from torch import nn
from torch.nn import functional as F

class SmoothedCrossEntropyLoss(nn.Module):
    '''

    '''

    def __init__(self,alpha=.154,reduce=True):
        super(SmoothedCrossEntropy,self).__init__()
        self.a = alpha
        self.reduce = reduce
        self.dist = nn.Softmax()
    
    def forward(self,inputs, targets):
        inputs = self.dist(inputs)
        out = torch.diag(inputs[:,targets]) * ((1.-self.a) + (self.a/float(inputs.size()[-1])))
        return -torch.mean(torch.log(out)) if self.reduce else -torch.log(out)

