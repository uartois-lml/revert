import torch
import plotly.express as px
import plotly.graph_objects as go

from torch.fft import rfft
from revert.transforms    import repeat

colors  = ['red', '#faa', 'blue', '#6af', '#381', '#6c9']
colorsV = ['purple', '#f6a', '#381', '#f83', '#fa8']

def scale_flows (x, fmt='torch'):
    m = x.mean(dim=[1])
    x = x * torch.sign(m)[:,None]
    x[2] *= (m[0] / m[2]).abs() 
    x[3] *= (m[1] / m[3]).abs()
    x[4] *= torch.sign(torch.dot(x[0], x[4]))
    return x 

def volumes (x, T=1):
    print(f"CC: {T} s")
    x = scale_flows(x) * 1e-6 * T / 32
    vb0 = (x[0] - x[2]).cumsum(0)
    vb1 = (x[1] - x[3]).cumsum(0)
    vcs = (x[4] - x[4].mean()).cumsum(0)
    vi0 = vb0 - vcs
    vi1 = vb1 - vcs
    v = torch.stack([vb0, vb1, vcs, vi0, vi1])
    return v - v.mean([1])[:,None]

def fig_vol (flows, Npulses=1, T=1):
    vol = volumes(flows, T)
    NT  = Npulses * T * 1e-3
    if Npulses > 1:
        vol = [repeat(Npulses)(v) for v in vol]    
    t = list(torch.linspace(0, NT, vol[0].shape[0]))
    y = [list(v) for v in vol]
    fig = px.line(x=t, y=y, 
        labels={'x': 'Time (s)', 'value': 'Volume change (mL)'},
        color_discrete_sequence=colorsV)
    fig.update_layout(
            showlegend=False, 
            margin=dict(l=30, r=30, t=30, b=30),
            height=300)
    return fig


def fig_flow (flows, Npulses=1, aqueduc=None, T=1): 
    NT = Npulses * T * 1e-3
    if Npulses > 1:
        return fig_flow(torch.stack(
            [repeat(Npulses)(f) for f in flows]), 
            aqueduc=repeat(Npulses)(aqueduc), T= Npulses*T)
    flows = scale_flows(flows) * 1e-3
    flows -= flows.mean([1])[:,None]
    x = list(torch.linspace(0, NT, flows.shape[1]))
    y = [list(f) for f in flows]
    if not isinstance(aqueduc, type(None)):
        aqueduc *= torch.sign(aqueduc * x[4])
        y += [list(aqueduc * 1e-3)] 
    fig = px.line(x=x, y=y, 
            labels={'x': 'Time (s)', 'value': 'Flow (mL / s)'},
        color_discrete_sequence=colors)
    fig.update_layout(
            showlegend=False, 
            margin=dict(l=30, r=30, t=30, b=30),
            height=300)
    return fig
