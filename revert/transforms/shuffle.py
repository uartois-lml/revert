import torch

def shuffle_all(x) :
    """
    Take all the data and shuffle each channel
    """
    dim = len(x.shape)
    if dim == 2 :
        x.unsqueeze_(0)
    Nc = x.shape[1]

    for i, xi in enumerate(x) :
        idx = torch.randperm(Nc)
        x[i] = xi[idx]
    
    if dim == 2 : 
        return x.squeeze(0)
    return x
        
        
def shuffle_two(x) :
    """
    Take all the data and take two random channel and swap them
    """
    dim = len(x.shape)
    if dim == 2 :
        x.unsqueeze_(0)
    Nc = x.shape[1]
    
    for i, xi in enumerate(x) :
        idx = torch.arange(Nc)
        rand1 = torch.randint(Nc, (1,1))
        rand2 = torch.randint(Nc, (1,1))
        idx[rand1] = rand2
        idx[rand2] = rand1
        x[i] = xi[idx]
        
    if dim == 2 : 
        return x.squeeze(0)
    
    return x