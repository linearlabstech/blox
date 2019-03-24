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

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

import os,yajl as json
import inspect
from importlib import import_module

def load(fname):
    modules = [fname["Module"]]#open(fname,'r').read()
    table = {}
    args = fname["Args"]
    for i,module in enumerate(modules):
        imported = import_module(module)
        # print(inspect.getmembers(imported))
        for c in inspect.getmembers(imported, inspect.isclass):
            # print(c[0],getattr(imported,c[0]))
            try:
                table[c[0]] = c[1](**args)
            except:
                pass
    return table

def GetLoss(f,kwargs):
    return getattr(nn,f)(**kwargs)

def GetOptim(models,f,kwargs):
    return getattr(optim,f)(models ,**kwargs)

def PrintSummary(tls,ttl,dls,tdl,e,s):
    print('\nEpoch Summary:')
    tls[e] = ttl
    dls[e] = tdl
    curr_pos = list('_'*s)
    curr_pos[e] = 'x'
    curr_pos = ''.join(curr_pos)
    print('\tRecent Training Loss: {}'.format(ttl))
    print('\t{}\n'.format(onelineplot(tls) ) )
    print('\tRecent Dev Loss: {}'.format(tdl))
    print('\t{}\n\t{}\n'.format(onelineplot(dls),curr_pos))

def onelineplot( x, chars=u" _▁▂▃▄▅▆▇█", sep="" ):
    """ numbers -> v simple one-line plots like
        Usage:
            astring = onelineplot( numbers [optional chars= sep= ])
        In:
            x: a list / tuple / numpy 1d array of numbers
            chars: plot characters, default the 8 Unicode bars above
            sep: "" or " " between plot chars

        How it works:
            linscale x  ->  ints 0 1 2 3 ...  ->  chars ▁ ▂ ▃ ▄ ...

        See also: https://github.com/RedKrieg/pysparklines
    """

    xlin = _linscale( x, to=[-.49, len(chars) - 1 + .49 ])
        # or quartiles 0 - 25 - 50 - 75 - 100
    xints = xlin.round().astype(int)
    assert xints.ndim == 1, xints.shape  # todo: 2d
    return sep.join([ chars[j] for j in xints ])


def _linscale( x, from_=None, to=[0,1] ):
    """ scale x from_ -> to, default min, max -> 0, 1 """
    x = np.asanyarray(x)
    m, M = from_ if from_ is not None \
        else [np.nanmin(x), np.nanmax(x)]
    if m == M:
        return np.ones_like(x) * np.mean( to )
    return (x - m) * (to[1] - to[0]) \
        / (M - m)  + to[0]