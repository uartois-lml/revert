import torch
import torch.nn as nn

def norm2 (t, dim=None):
    """ L2-norm on specified dimensions """
    return torch.sqrt((t ** 2).sum(dim)) 

def cross_correlation (ya, yb):
    """ Cross-correlation of N_batch x N """
    yab = ya[:,:,None] @ yb[:,None,:]
    return yab.sum(dim=[0]) / (norm2(ya, [0]) * norm2(yb, [0]))

class BarlowTwins (nn.Module):

    def __init__(self, model, offdiag=0.5):
        """ Create twins from a model. """
        super().__init__()
        self.model   = model
        self.offdiag = offdiag 

    def forward (self, x):
        """ Apply twins to 2 x N_batch x N tensor. """
        xa, xb = x
        ya, yb = self.model(xa), self.model(xb)
        return torch.stack([ya, yb])

    def loss (self, y): 
        """ Return Barlow twin loss of N_batch x N output. """
        n_out = y.shape[-1]
        C = cross_correlation(*y) 
        I = torch.eye(n_out)
        lbda      = self.offdiag
        loss_mask = lbda * torch.ones(C.shape) + (1 - lbda) * I
        return torch.sum(((C - I) * loss_mask) ** 2) / (2 * n_out ** 2) 

    def fit (self, x, lr=1e-2, br=1e-3, n_batch=256):
        """ Fit on a 2 x N_samples x N tensor. """
        n_it = x.shape[1] // n_batch
        for s in range(n_it):
            y = self.forward(x[:,s:s + n_batch])
            loss = self.loss(y)
            loss.backward()
            with torch.no_grad(): 
                for p in self.parameters(): 
                    p -= p.grad * lr
                    p -= br * torch.randn(p.shape)
                self.zero_grad()
            return self



        
         

