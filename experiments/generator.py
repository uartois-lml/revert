from revert.transforms import shift_all
from revert.models import ConvNet, Module, Pipe

from torch.utils.data import TensorDataset, DataLoader

import torch 

# ============== Data =============================

def getData(stdev) :
    data = torch.load("../scripts-pcmri/pcmri_tensor.pt")
    flows = data["flows"]

    shifted, y = shift_all(stdev)(flows)

    data_dataset = TensorDataset(shifted, y)
    data_loader = DataLoader(data_dataset, shuffle=True, batch_size=1)
    
    return data_loader


#==================================================

#--- Models ---

def getModel(Npts = 32) :
    layers = [[Npts, 6,   8],
              [16,  6*12,  8],
              [8,   6*24,  8],
              [1,   6*12,  1]]

    base = ConvNet(layers, pool='max')

    dim_out = 6*12
    dim_task = 6
    head = ConvNet([[1, dim_out, 1], [1, dim_task, 1]]) 
    
    #--- loss function ---
    def mse_loss(y, y_tgt):
        return ((y - y_tgt)**2).sum()
    
    head.loss = lambda y, y_tgt : mse_loss(y, y_tgt)
    
    return Pipe(base, head)