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
from torch import nn
import torch

class Recurrence(nn.Module):
    '''
        This is meant for RNN integration in BLOX
    '''

    def __init__(self):
        super(Recurrence,self).__init__()

    def __call__(self,x):
        (y,h) = x
        # return (y).squeeze(1)
        # y = y.view(len(y), -1)
        # print(y.size())
        return y
