import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash import callback_context
import plotly.express as px

import torch
import json
from os import path 
from glob import glob
from graphs import fig_flow, fig_vol

from revert import pcmri

#--- Flow pulses ---
db = pcmri.Dataset('full')

#--- State ---
global keys
keys    = db.ls()
global index
tag     = "good" 
index   = 1 

#--- app ---
app = dash.Dash(__name__)

#--- Controls ---

controls = html.Div(id="controls", children=[
    html.Div(children=[
        html.Span(children='Pulses: '),
        dcc.Input(id='n-pulses', value=3)
    ])
])

tags = html.Div(id="tags", children=[
    html.Button(id="good", children='Good'),
    html.Button(id="noisy", children='Noisy'),
    html.Button(id="bad", children='Bad')
])
seek = html.Div(id="seek", children=[
    html.Button(id="prev", children="<"),
    html.Button(id="next", children=">")])
btns = html.Div(id="btns", children=[tags, seek])

keydiv = html.Div(id='key', children=f'b*')
agediv = html.Div(id='age', children='-1')
tagdiv = html.Div(id='tag', children='good')
info = html.Div(id="info", children=[tagdiv, keydiv, agediv])

#--- Layout --- 

app.layout = html.Div(children=[
    html.Div(className="flex-h", 
             children=[controls, btns, info]),
    dcc.Graph(id='segments'),
    dcc.Graph(id='volumes')
])

#--- Callbacks ---

@app.callback(
    Output('segments', 'figure'),
    Output('volumes', 'figure'),
    Output('age', 'children'),
    Input('key', 'children'),
    Input('n-pulses', 'value'))
def update_exam(key, n):
    file  = db.get(key)
    T = file.read('c2-c3', ['time'])[0][-1]
    print(file)
    flows = file.flows()
    flows *= torch.sign(flows.mean(dim=[1])[:,None])
    aq = file.read('aqueduc')[0]
    age = file.age()
    print(age)
    try:
        Npulses = int(n) 
    except:
        Npulses = 1
    return [fig_flow(flows, Npulses, aqueduc=aq, T=T), 
            fig_vol(flows, Npulses, T),
            f'age: {age}']

@app.callback(
    Output('tag', 'children'),
    Input('good', 'n_clicks'),
    Input('noisy', 'n_clicks'),
    Input('bad', 'n_clicks'))
def tag_exam(n1, n2, n3):
    changed = [p['prop_id'] for p in callback_context.triggered][0]
    print(f"changed: {changed}")
    tag = changed.split(".")[0]
    tag = tag if tag in ["good", "noisy", "bad"] else "good"
    return 'good'

@app.callback(
    Output('key', 'children'),
    Input('prev', 'n_clicks'),
    Input('next', 'n_clicks'))
def seek_exam(p, n):
    changed = [p['prop_id'] for p in callback_context.triggered][0]
    print(f"changed: {changed}")
    btn = changed.split(".")[0]
    if btn == "prev":
        mv = -1
    elif btn == "next":
        mv = +1
    else:
        mv = 0
    global keys
    global index
    index += mv
    return keys[(index + mv) % len(keys)]

#--- Run on 8050 ---

if __name__ == '__main__':
    app.run_server(debug=True)
