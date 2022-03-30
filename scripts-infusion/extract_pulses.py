import torch
import os
import json
import tqdm

from revert import infusion
from revert.models     import ConvNet
from revert.transforms import filter_spikes, bandpass, Troughs, diff
from revert.transforms import segment, mask_center

dbname = "full"
dest   = "baseline.pt"
fs = 100

db = infusion.Dataset(dbname)

def main (model_state="pretrained.pt", Npulses=64, minutes=6): 

    # model variation losses
    print(f"loading model from '{model_state}'")
    model = ConvNet.load(model_state)
    losses = [mean_loss(model), diff_loss(model)]
    loss = mixed_loss(losses)

    # files with baseline
    print(f"filtering files with 'Baseline' timestamps")
    keys = db.filter(lambda f: db.periods[f.key]["Baseline"])

    # filters and peak detection 
    print(f"extracting pulses from {len(keys)} recordings")
    Npts    = minutes * 6000
    bp      = bandpass(.6, 12, fs)
    argmin  = Troughs(Npts, 50)

    # main loop
    out = []
    errors = []
    for k in tqdm.tqdm(keys): 
        evts = db.periods[k]
        file = db.get(k)
        try:
            i0 = int(100 * (evts["Baseline"][0] - evts["start"]))
            icp = file.icp(i0, Npts)
            icp = filter_spikes(icp)[0]
            troughs = argmin(bp(icp))
            if icp.shape[0] != Npts:
                raise RuntimeError("not enough points")
            out += [select_pulses(bp(icp), troughs, Npulses, loss)]
        except:
            errors += [k]
        file.close()

    # save output
    print(f"saving output as '{dest}'")
    xs    = [torch.stack([x[i] for x in out]) for i in range(4)]
    names = ["masks", "pulses", "means", "slopes"]
    data  = {ni: xi for xi, ni in zip(xs, names)}
    for n in names: 
        print(f"  + {n}\t: {list(data[n].shape)} tensor")
    data["keys"]   = [k for k in keys if not k in errors] 
    data["errors"] = errors
    torch.save(data, f'{dest}')
    # save keys
    print(f"extracted {Npulses} pulses from {len(data['keys'])} recordings")
    print(f"encountered {len(errors)} errors")


def select_pulses (icp, troughs, Npulses, loss):
    Npts = icp.shape[-1]
    # find diastoles
    # segment and center pulses
    pulses, mask = segment(icp, troughs, 128)
    x, x_mean, x_slope = mask_center(pulses, mask, output='slopes')
    # sort by loss value
    z = loss(x)
    idx = z.sort().indices[:Npulses]
    return mask[idx], x[idx], x_mean[idx], x_slope[idx]

def mean_loss(model=None):
    def loss(x):
        with torch.no_grad():
            y = (model(x) if not isinstance(model, type(None)) 
                          else x)
        y_mean = y - y.mean([0])[None,:]
        return y_mean.norm(dim=[-1])
    return loss

def diff_loss(model=None):
    def loss(x):
        with torch.no_grad():
            y = (model(x) if not isinstance(model, type(None)) 
                          else x)
        dy = diff(y.T).T
        return dy.norm(dim=[-1])
    return loss

def mixed_loss(losses, weights=None):
    if isinstance(weights, type(None)):
        weights = torch.ones([len(losses)])
    if not isinstance(weights, torch.Tensor): 
        weights = torch.tensor(weights)
    def loss(x):
        z = torch.stack([l(x) for l in losses])
        return (z * weights[:,None]).sum([0]) / weights.sum()
    return loss

if __name__ == '__main__':
    main()
