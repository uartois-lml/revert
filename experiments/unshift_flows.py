from revert.models import ConvNet, Module
from experiments import arg_parser, read_args, getModel, getData

import json

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

import torch

# model state 

#--- Main ---

def main(defaults=None, stdev=0.5):
    
    if defaults is None : 
        defaults = {'epochs':  5,
                    'n_batch': 128,
                    'lr':      1e-3,
                    'gamma':   0.8,
                    'n_it':    3750,
                    'stdev' : stdev
                    } | (defaults if defaults else {})
    else : 
        defaults = defaults | { 'stdev' : stdev }

    # take all the data
    dataLoad = getData(defaults['stdev']) 
    # generate the model
    model = getModel()
    # find the path to save 
    args = arg_parser(prefix = 'convnet')
    path = read_args(args, prefix = 'convnet')
        
    print(model.modules)
    #--- optimizer ---
    optim = Adam(model.parameters(), lr=defaults['lr'])
    lr    = ExponentialLR(optim, gamma=defaults['gamma'])
        
    model.writer = SummaryWriter(path.writer)
    
    model.fit(dataLoad, optim, lr, epochs=defaults['epochs'], w="Loss")
    free(optim, lr)
    
    # save the hyper parameter to tensorboard
    for key, value in defaults.items():
        model.writer.add_text(key , str(value))
    
    #save the model
    print(path.output)
    print(model.modules[-1])
    print(model.modules[-1].state_dict())
    model.save(path.output)
            
#--- Cuda free ---

def free (*xs):
    for x in xs: del x
    torch.cuda.empty_cache()

def different_test():
    main()
    
    defaults = {'epochs':  5,
                    'n_batch': 128,
                    'lr':      1e-3,
                    'gamma':   0.8,
                    'n_it':    3750
                    }
    main(defaults)
    
    defaults = {'epochs':  5,
                'n_batch': 256,
                'lr':      1e-3,
                'gamma':   0.8,
                'n_it':    3750
                }
    main(defaults, 0.5)


different_test()