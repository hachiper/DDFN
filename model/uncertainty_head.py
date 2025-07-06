import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch import log, pi, lgamma



# Negative Log Likelihood
# Prior: Normal Inverse-Gamma (NIG) distribution
def NIG_NLL(y, gamma, nu, alpha, beta):
    omega = 2*beta*(1 + nu)
    nig_nll = 0.5*log(pi/nu) - alpha*log(omega) + (alpha + 0.5)*log((y - gamma)*(y - gamma)*nu + omega) + lgamma(alpha) - lgamma(alpha + 0.5)
    return nig_nll.mean()

# A penalty whenever there is an error in the prediction
# Scales with the total evidence of our inferred posterior
def NIG_Regularization(y, gamma, nu, alpha):
    # error = (y - gamma)*(y - gamma)
    error = torch.abs(y - gamma)
    evidence = 2 * nu + alpha
    return (error*evidence).mean()
# NLL loss = alpha*log(beta)-log(T(alpha))+0.5 * log(2 * pi * nu ** nu) + (1 + alpha)*log(nu**2)+(2*beta+nu*(y-gamma)**2)/(2*nu*nu)

class UncertaintyHead(nn.Module):
    
    def __init__(self, input_dim,stride):
        super(UncertaintyHead, self).__init__()
        self.epsilon = 2.5e-2
        self.max_rate = 1e-4
        self.lambda_coef = 0 
        self.stride = stride
        
        self.MLP = nn.Linear(input_dim * 3, 4*stride)
        self.linear = nn.Linear(4*stride, 4*stride)
        # self.act = nn.LeakyReLU()
        # self.bn = nn.BatchNorm1d(4*stride)
        # self.act = nn.ReLU()
        self.act = nn.Sigmoid()
        # self.act = nn.Tanh()
    
    def forward(self, x):
        out = self.act(self.MLP(x))
        # out = self.bn(out)
        # out = self.act(self.linear(out))        
        # out = self.act(self.bn(self.MLP(x)))

        out = out.reshape(out.shape[0],self.stride,-1)
        # print(out.shape)
        gamma, nu, alpha, beta = torch.split(out, 1, dim=2)
        nu, alpha, beta = fun.softplus(nu), fun.softplus(alpha), fun.softplus(beta)
        return gamma, nu, alpha, beta
    
    def get_loss(self, y, gamma, nu, alpha, beta):
        nig_loss, nig_regularization = NIG_NLL(y, gamma, nu, alpha, beta), NIG_Regularization(y, gamma, nu, alpha)
        # loss = nig_loss + (nig_regularization - self.epsilon)*self.lambda_coef
        loss = nig_loss + nig_regularization
        return loss, nig_loss, nig_regularization
    
    def hyperparams_update(self, nig_regularization):
        with torch.no_grad():
            self.lambda_coef += self.max_rate * (nig_regularization - self.epsilon)
