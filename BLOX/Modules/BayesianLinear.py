
import torch,math
from torch import nn
from torch.nn import functional as F
from BLOX.Modules.Gaussian import Gaussian
from BLOX.Modules.GMM import GMM
class BayesianLinear(nn.Module):
    '''
        Linear Layer for Bayesian Neural Networks
    '''
    def __init__(self, in_features, out_features,n_mixtures=2):
        super(BayesianLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        pis = [float(1./n_mixtures)]*n_mixtures
        sigmas = [ torch.FloatTensor([math.exp(-0)]).to( 'cuda' if torch.cuda.is_available() else 'cpu' ) for m in range(n_mixtures)]

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = GMM(pis, sigmas)
        self.bias_prior = GMM(pis, sigmas)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior(weight) + self.bias_prior(bias)
            self.log_variational_posterior = self.weight(weight) + self.bias(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
            
        return F.linear(x, weight, bias)