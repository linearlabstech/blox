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

from torch import nn

# from .DEFS import *

def handle_linear(kwargs):
    return nn.Linear(**kwargs)

def handle_act(act):
    if hasattr(nn,act):
        return getattr(nn,act)()

def handle_(args):
    return nn.Sequential(*[getattr(nn,list(n.keys())[0])(list(n.values())[0]) for n in args])

handlers = {
    "Linear":handle_linear,
    "Act":handle_act,
    "Other":handle_
}
from ..Modules.PlaceHolder import PlaceHolder

ph = PlaceHolder()

print(ph)

def Compile(obj):
    if isinstance(obj,str):
        assert obj.endswith('.json'),'Config type not currently supported'
        obj = json.load(open(obj,'r'))
    layers = []
    COMPONENTS = {'PlaceHolder':ph}
    if 'IMPORTS' in obj:
        for f in obj['IMPORTS']: #COMPONENTS.update( Compile(json.load(open(f,'r'))["BLOCKS"] ) )if isinstance(f,str) else 
            if isinstance(f,dict):
                COMPONENTS.update( load( f ) )
    for k,fs in obj['DEFS']['BLOX'].items():COMPONENTS.update( {k: nn.Sequential(*[handlers[k](v) for f in fs for k,v in f.items() ]) } )
    for layer in obj['BLOX']:
        if 'DEF' in layer:
            f = layer['DEF']
            # if the block is defined in another file,load and continue
            if f.endswith('.json') or f.endswith('.blx'):
                funcs = Compile( json.load( open(f,'r') ) )
            # check to see if the block is previously defined
            elif f in COMPONENTS:
                funcs = COMPONENTS[f] 
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
        

if __name__ == '__main__':
    print( Compile( json.load(open( 'net.json', 'r') ) ) )



