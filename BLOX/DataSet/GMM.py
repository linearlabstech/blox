
import torch
from torch import nn
import math
CONST = math.sqrt( 2*math.pi )

class GF(torch.autograd.Function):

     @staticmethod
    def forward(ctx, x, mu, sigma):
        ctx.save_for_backward(input, weight, bias)
        return ( 1.0/ (sigma * CONST) )*torch.exp(-.5*( torch.pow( (x-mu)/sigma,2.0 ) )  )

    @staticmethod
    def backward(ctx, grad_output):
        x,mu,sigma = ctx.saved_tensors
        grad_x = grad_mu = grad_sigma = None

        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0).squeeze(0)

        # return grad_input, grad_weight, grad_bias
        if ctx.needs_input_grad[1]: grad_mu = (x-mu)/float(len(mu))
        if ctx.needs_input_grad[2]:
            invsig = torch.inverse(sigma)
            grad_sigma = 0.5 * (-invsig + torch.sum(invsig.dot(x - mu) * x))

        return grad_x,grad_mu, grad_sigma 



class Guassian(nn.Module):

    def __init__(self,input_size,mu=None,sigma=None):

        if mu is None:
            mu = torch.zeros(input_size)
        if sigma is None:
            sigma = torch.ones(input_size)
        self.mu = mu
        self.sigma = sigma
    
    def forward(self,x):
        return GF.apply(x,self.mu,self.sigma)