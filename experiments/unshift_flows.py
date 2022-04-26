from signal import default_int_handler
from revert.transforms import noise, vshift, scale, shift_all
from revert.models import ConvNet, Module

import argparse, sys, os
import datetime
import json

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
import torch

# ============== Data =============================

def getData(stdev) :
    data = torch.load("../scripts-pcmri/pcmri_tensor.pt")
    flows = data["flows"]

    global shifted
    shifted, y = shift_all(stdev)(flows)

    batch_size = 8

    data_dataset = TensorDataset(shifted, y)
    data_loader = DataLoader(data_dataset, shuffle=True, batch_size=batch_size)
    
    return data_loader


#==================================================

#--- Models ---
# Need to be ajust with the two heads

def getModel() :
    global Npts
    Npts = shifted.shape[2]
    layers = [[Npts, 6,   8],
              [16,  6*12,  8],
              [8,   6*24,  8],
              [1,   6,  1]]

    model = ConvNet(layers, pool='max')

    dim_out = 6*12
    dim_task = 6
    head1 = ConvNet([[1, dim_out, 1], [1, dim_task, 1]]) 
    head2 = ConvNet([[1, dim_out, 1], [1, dim_task, 1]]) 
    
    return model, head1, head2

#===== State dict / logdir as CLI arguments ===== 
#
#   python pcmri_unshift.py -s "model.state" -w "runs/pcmri_unshift-xx"

parser = argparse.ArgumentParser()
parser.add_argument('--state', '-s', help="load state dict", type=str)
parser.add_argument('--writer', '-w', help="tensorboard writer", type=str)
args = parser.parse_args()

# model state 
# generate the name of the differents folder/files

def generate_files():
    # generate the month name
    datem = datetime.datetime.now()
    month_num = str(datem.month)
    datetime_object = datetime.datetime.strptime(month_num, "%m")
    month_name = datetime_object.strftime("%b").lower()
    
    # get the day number
    global day_number 
    day_number = str(datem.day)

    global name 
    global modelExist 
    modelExist = False
    if args.state: 
        print(f"Loading model state from '{args.state}'")
        if os.path.exists(args.state) :
            model.load(args.state)
            modelExist = True
        else : 
            print("The model doesn't exist, a new one will be generated")
            name = month_name
    else : 
        name = month_name

    # writer name
    log_dir = args.writer if args.writer else None
    if log_dir: 
        model.writer = SummaryWriter("runs/convnet-"+log_dir)
    else :
        # generate the number of the file
        global number
        number = 1
        while ( os.path.exists("runs/convnet-"+month_name+day_number+"-"+str(number)) 
               or os.path.exists("models/convnet-"+name+day_number+"-"+str(number)+".pt") ) :
            number+=1
        number = str(number)
        model.writer = SummaryWriter("runs/convnet-"+month_name+day_number+"-"+number)

#--- Main ---

def main(defaults=None, stdev=0.5):
    
    if defaults is None : 
        defaults = {'epochs':  20,
                    'n_batch': 128,
                    'lr':      1e-3,
                    'gamma':   0.8,
                    'n_it':    3750,
                    'stdev' : stdev
                    } | (defaults if defaults else {})
    else : 
        defaults = defaults | { 'stdev' : stdev }

    dataLoad = getData(defaults['stdev']) 
    global model
    global head1
    global head2
    model, head1, head2 = getModel()
    generate_files()
    
    #--- optimizer ---
    optim = Adam(model.parameters(), lr=defaults['lr'])
    lr    = ExponentialLR(optim, gamma=defaults['gamma'])

    #--- loss function ---
    def mse_loss(y, y_tgt):
        return ((y - y_tgt)**2).sum()
    
    model.Npts = Npts
    model.loss = lambda y, y_tgt : mse_loss(y, y_tgt)
    
    model.head1 = head1
    model.head2 = head2
    
    model.fit(dataLoad, optim, lr, epochs=defaults['epochs'], w="Loss")
    free(optim, lr)
    
    # save the hyper parameter to tensorboard
    model.writer.add_text("epochs : ", str(defaults['epochs']))
    model.writer.add_text("n_batch : ", str(defaults['n_batch']))
    model.writer.add_text("lr : ", str(defaults['lr']))
    model.writer.add_text("gamma : ", str(defaults['gamma']))
    model.writer.add_text("n_it : ", str(defaults['n_it']))
    model.writer.add_text("stdec : ", str(defaults['stdev']))

    #save the hyper parameter in a json file
    with open('runs/json-'+name+day_number+"-"+number+'.json', 'w') as js:
        json.dump(defaults, js)

    # save the model
    if modelExist : 
        model.save(args.state)
    else :
        model.save("models/convnet-"+name+day_number+"-"+number+".pt")
    #head1.save("models/head1-"+name+day_number+"-"+number+".pt")
    #head2.save("models/head2-"+name+day_number+"-"+number+".pt")
            
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