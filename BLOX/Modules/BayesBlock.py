
import torch
from torch import nn
from torch.nn import functional as F

from BLOX.Modules.BayesianLinear import BayesianLinear
from BLOX.Modules.Sparsemax import Sparsemax
class BayesBlock(nn.Module):
    '''
        you can either pass in a compute block of bayesian layers or construct a list of layers all the same size
        The reason you'd want to pass a custome block would be to use different sizes and tricks.
        basically do a resnet type of procedure where you add the input at the end
    '''

    def __init__(self,in_features,out_features,act='ReLU',depth=3,sparse=True,n_samples=4,block=None):
        super(BayesBlock,self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.n_smpl = n_samples
        self.layers = [ BayesianLinear(in_features,out_features) for _ in range(depth) ] if block == None else block
        if torch.cuda.is_available():
            for l in self.layers:l.cuda()
        self.act = getattr(nn,act)()
        self.out = Sparsemax() if sparse else nn.Softmax()


    def log_prior(self):
        return (l.log_prior for l in self.layers ).sum()
    
    def log_variational_posterior(self):
        return (l.log_variational_posterior for l in self.layers ).sum()

    def sample(self,x):
        
        for l in self.layers:
            x = l(x,sample=True)
            
            x = self.act(x)
        return self.out(x)

    def forward(self,x):
        '''
            assume batch first
        '''
        if len(x.size()) == 2:x = x.unsqueeze(0)
        dists = torch.zeros(x.size()[0],self.n_smpl ,self.n_out).to('cuda' if torch.cuda.is_available() else 'cpu')
        # log_probs = torch.zeros(self.n_smpl).to('cuda' if torch.cuda.is_available() else 'cpu')
        # log_variational_posteriors = torch.zeros(self.n_smpl).to('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(self.n_smpl):
            dists[:,i] = self.sample(x)
            # log_probs[i] = self.log_prior()
            # log_variational_posteriors[i] = self.log_variational_posterior()
        return dists.mean(1) + x

