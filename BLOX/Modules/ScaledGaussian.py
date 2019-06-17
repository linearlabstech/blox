
import torch

class GMM(torch.nn.Module):
    def __init__(self, pis, sigmas):
        super().__init__()
        self.pis = pis
        self.sigmas = sigmas
        self.mixtures = [torch.distributions.Normal(0,sigma) for sigma in sigmas]
    
    def forward(self, x):
        p1 = [  torch.exp( g(x) ) for g in self.mixtures ]
        return ( torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()

    def backward(self):pass