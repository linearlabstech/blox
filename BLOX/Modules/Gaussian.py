import torch,math


class Gaussian(torch.nn.Module):
    def __init__(self, mu, rho):
        super(Gaussian,self).__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to( 'cuda' if torch.cuda.is_available() else 'cpu' )
        return self.mu + self.sigma * epsilon
    
    def forward(self, x):
        return (-math.log(math.sqrt(2 * 3.14156))
                - torch.log(self.sigma)
                - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()