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
op_table = {
    '*':'__mul__',
    '/':'__div__',
    '+':'__add__',
    '-':'__sub__'
}
class PlaceHolder(nn.Module):
    X = None
    gate = True
    op = ''
    def __init__(self,operator='+'):
        super(PlaceHolder,self).__init__()
        self.op = operator

    def __call__(self,X):
        self.gate = not self.gate
        if not self.gate:
            self.X = X.clone()
            return X
        else:
            Y = self.X
            self.X = None
            return getattr(X,op_table[self.op])(Y) if self.op in op_table else Y
        