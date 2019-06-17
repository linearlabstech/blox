
import torch,math

from BLOX.Modules.Gaussian import Gaussian

class GMM(torch.nn.Module):
    def __init__(self, pis, sigmas):
        super(GMM,self).__init__()
        self.pis = pis
        self.sigmas = sigmas
        self.mixtures = [Gaussian(0,sigma) for sigma in sigmas]
    
    def forward(self, x):
        probs = [  torch.exp( g(x) ) for g in self.mixtures ]
        return torch.log( sum([ pi*prob for pi,prob in zip(self.pis,probs) ] ) )
