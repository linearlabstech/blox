#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, Linear Labs Technology

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
from ..Common.utils import StrOrDict as sod,load
from torch import nn
class PreProcessor(nn.Module):
    processors = []
    def __init__(self,cfg):
        super(PreProcessor,self).__init__()
        cfg = sod(cfg)
        self.steps = load(cfg['IMPORTS'])
        self.order = cfg['BLOX']

    def forward(self,x):
        for o in self.order: x = 


        