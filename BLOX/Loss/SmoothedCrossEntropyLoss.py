

import torch
# from torch.nn.modules.loss import _WeightedLoss
from torch import nn
from torch.nn import functional as F
from BLOX.Modules.Sparsemax import Sparsemax
class SmoothedCrossEntropyLoss(nn.Module):
    '''

    '''

    def __init__(self,alpha=.154,reduce=True):
        super(SmoothedCrossEntropyLoss,self).__init__()
        self.a = alpha
        self.reduce = reduce
        self.dist = nn.Softmax(dim=-1)
    
    def forward(self,inputs, targets):
        inputs = self.dist(inputs)
        t = torch.ones_like(inputs)
        out = torch.diag(t[:,targets])  * ((1.-self.a) + (self.a/float(inputs.size()[-1])))
        return -torch.sum( inputs * torch.log(out - inputs)) if self.reduce else -torch.log(out)
