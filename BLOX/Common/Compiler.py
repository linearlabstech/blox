#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, Linear Labs Technologies

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import yajl as json
import torch
from torch import nn
from BLOX.Common import Functions as blx_fn
from ..Common.utils import load,load_dynamic_modules,StrOrDict as sod
from BLOX.Modules import *
# from BLOX.Common.Globals import *
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
def handle_linear(kwargs):
    return nn.Linear(**kwargs)

def handle_act(act):
    if hasattr(nn,act):
        return getattr(nn,act)()
    elif hasattr(blx_fn,act):return getattr(blx_fn,act)()
    else: raise AttributeError("No activation function exists {}".format(act))

def handle_(args):
    return nn.Sequential(*[getattr(nn,k)(**v) for n in args for k,v in n.items()])
def handle_bayes(kwargs):
    return BayesianLinear.BayesianLinear(**kwargs)

def handle_bayes_block(kwargs):
    return BayesBlock.BayesBlock(**kwargs)

handlers = {
    "Linear":handle_linear,
    "Act":handle_act,
    "Other":handle_,
    "BayesianLinear":handle_bayes,
    "BayesBlock":handle_bayes_block
}


class Wrapper(nn.Module):
    '''
        a simple wrapper class around multiple networks
    '''

    def __init__(self,order=None,nets=None,cfg=None):
        super(Wrapper,self).__init__()
        self.order = []
        self.nets = {}
        if not cfg and (not order or nets):raise ValueError('You have to either pass a config or the order and networks to the Wrapper class')
        if cfg:
            cfg = sod(cfg)
            for k,v in cfg['Nets'].items():
                self.order.append(k)
                self.nets[k] = Compile( json.loads(open(cfg['Nets'][k],'r').read() ) )
            if "Load" in cfg:
                try:self.load_state_dict(cfg['Load'])
                except:pass
        else:
            self.order = order
            self.nets = nets
        self.likelihood = None
        for o in self.order:
            if hasattr(self.nets[o],'likelihood'):self.likelihood = self.nets[o].likelihood

    def __repr__(self):
        return '\n'.join( repr(self.nets[o]) for o in self.order )
    
    def __str__(self):
        return self.__repr__()
    
    def forward(self,x):
        for o in self.order: 
            x = self.nets[o](x)
        return x

    def load_state_dict(self,states):
        states = sod(states)
        for k,v in states.items():
            try:
                if isinstance(v,str):
                    try:self.nets[k].load_state_dict(torch.load(v))
                    except:self.nets[k].load_state_dict(torch.load(v,map_location='cpu'))
                else:self.nets[k].load_state_dict(v)
            except: raise Exception('Unable to load block {}'.format(k))
        return self



def cfg2nets(cfg):
    return Wrapper(cfg=cfg)
PREDEFINED = load_dynamic_modules('BLOX.Modules')

def Compile(obj):
    if isinstance(obj,str):
        assert obj.endswith('.json'),'Config type not currently supported'
        obj = json.load(open(obj,'r'))
    layers = []
    COMPONENTS = {}
    if 'IMPORTS' in obj:
        assert isinstance(obj['IMPORTS'],list), 'BLOX IMPORTS must be a string'
        for f in obj['IMPORTS']:
            if isinstance(f,str):
                if f.endswith('.blx') or f.endswith('.json'):
                    _obj = json.load( open(f,'r') ) 
                    obj_name = _obj['Name'] if 'Name' in _obj else  ''.join(f.split('.')[:-1])
                    COMPONENTS.update( 
                            { 
                                obj_name:Compile( _obj ) 
                            } 
                        )
                else: 
                    COMPONENTS.update( load( f )  )
                    # USER_DEFINED.update( load( f )  )
    if 'DEFS' in obj:
        for k,fs in obj['DEFS'].items():COMPONENTS.update( {k: nn.Sequential(*[handlers[k](v) for f in fs for k,v in f.items() ]) } )
    if 'BLOX' in obj:
        for layer in obj['BLOX']:
            if isinstance(layer,str):                
                f = layer
                print(layer,json.dumps(list(PREDEFINED.keys()),indent=2))
                if f in COMPONENTS:
                    funcs = COMPONENTS[f]
                elif f in PREDEFINED:
                    funcs = PREDEFINED[f]()
                else:
                    raise ValueError('Function Block not defined in config file. Error @ {}'.format(f))
                layers.append( funcs )
            else:
                if 'MODULE' in layer:
                    layers.append(PREDEFINED[layer['MODULE']['LAYER']](**layer['MODULE']['ARGS']) if isinstance(layer['MODULE'],dict) else PREDEFINED[layer['MODULE']]() )
                elif 'DEF' in layer:
                    f = layer['DEF']
                    # if the block is defined in another file,load and continue
                    if f.endswith('.json') or f.endswith('.blx'):
                        funcs = Compile( json.load( open(f,'r') ) )
                    # check to see if the block is previously defined
                    elif f in COMPONENTS:
                        funcs = COMPONENTS[f]
                    elif f in PREDEFINED:
                        funcs = PREDEFINED[f]()
                    else:
                        raise ValueError('Function Block not defined in config file. Error @ {}'.format(f))
                    layers.append( funcs )
                elif 'REPEAT' in layer:
                    b = layer['REPEAT']['BLOCK']
                    t = layer['REPEAT']['REPS']
                    layers.append( nn.Sequential(*[ c for _ in range(t) for c in COMPONENTS[b] ]) if isinstance(COMPONENTS[b],list) else COMPONENTS[b] ) 
                else:
                    for k,v in layer.items():
                        layers.append( handlers[k](v) )
    return nn.Sequential(*layers).cuda() if torch.cuda.is_available() else nn.Sequential(*layers)
        



