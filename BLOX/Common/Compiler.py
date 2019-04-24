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
from ..Common.utils import load,load_dynamic_modules

def handle_linear(kwargs):
    return nn.Linear(**kwargs)

def handle_act(act):
    if hasattr(nn,act):
        return getattr(nn,act)()

def handle_(args):
    return nn.Sequential(*[getattr(nn,k)(**v) for n in args for k,v in n.items()])

handlers = {
    "Linear":handle_linear,
    "Act":handle_act,
    "Other":handle_
}
from BLOX.Modules import *


def cfg2nets(cfg):
    nets = {}
    order = []
    for k,v in cfg['Nets'].items():
        order.append(k)
        nets[k] = Compile( json.loads(open(cfg['Nets'][k],'r').read() ) )
    if "Load" in cfg:
        for k,v in cfg['Load'].items():
            try:
                nets[k].load_state_dict(torch.load(v))
            except:pass
    return order,nets
PREDEFINED = load_dynamic_modules('BLOX.Modules')

def Compile(obj):
    if isinstance(obj,str):
        assert obj.endswith('.json'),'Config type not currently supported'
        obj = json.load(open(obj,'r'))
    layers = []
    COMPONENTS = {}
    if 'IMPORTS' in obj:
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
    if 'DEFS' in obj:
        if 'BLOX' in obj['DEFS']:
            for k,fs in obj['DEFS']['BLOX'].items():COMPONENTS.update( {k: nn.Sequential(*[handlers[k](v) for f in fs for k,v in f.items() ]) } )
    if 'BLOX' in obj:
        for layer in obj['BLOX']:
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
                    raise NotImplementedError('Function Block not defined in config file. Error @ {}'.format(f))
                layers.append( funcs )
            elif 'REPEAT' in layer:
                b = layer['REPEAT']['BLOCK']
                t = layer['REPEAT']['REPS']
                layers.append( nn.Sequential(*[ c for _ in range(t) for c in COMPONENTS[b] ]) if isinstance(COMPONENTS[b],list) else COMPONENTS[b] ) 
            else:
                for k,v in layer.items():
                    layers.append( handlers[k](v) )
    return nn.Sequential(*layers)
        



