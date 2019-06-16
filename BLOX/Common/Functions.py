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
from torch import nn
import math


class Swish(nn.Module):
    """ Swish activation function """
    def __init__(self):
        super(Swish,self).__init__()

    def forward(self,x):return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GELU(nn.Module):
    """
        Implimentaiton od GAUSSIAN ERROR LINEAR UNITS
        see: https://arxiv.org/pdf/1606.08415.pdf for details

        I mean, in the paper they state they want a bernoulli distribution on droping connections, 
        but we can just use the below formulation to approximate.
    """
    def __init__(self):super(GELU,self).__init__()

    def forward(self,x):  x * torch.sigmoid( 1.702*x )