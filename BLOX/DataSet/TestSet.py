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
from random import shuffle
class TestSet:
    """
        load a saved pt state of shape:

        [
            [input,output],
            ...,
        ]
        
        TODO: add support for testing split as well

    """
    size = 0
    n_classes = -1
    data = None
    _data = None
    def __init__(self,data,eval_split=.15,dtype='float'):
        self._data = data if isinstance(data,dict) else torch.load(data)
        size = len(self._data['inputs'])
        self.type = dtype
        self.size = size
        self.idxs = list(range(self.size))
        if torch.cuda.is_available():self.cuda()

    def __len__(self):return self.size
    
    def shuffle(self):
        shuffle(self.idxs)
        return self
    
    def cuda(self):
        """
            Move data GPU
        """
        for i in range(self.size):
            self._data['inputs'][i] = self._data['inputs'][i].cuda()
            self._data['targets'][i] = self._data['targets'][i].cuda()

    def cpu(self):
        """
            Move data CPU
        """
        for i in range(self.tsize):
            self._data['inputs'][i] = self._data['inputs'][i].cpu()
            self._data['targets'][i] = self._data['targets'][i].cpu()


    def to(self, dtype):
        dtype = dtype.lower()
        assert dtype in ['cpu','gpu'], 'data type not supported'
        self.cpu() if dtype == 'cpu' else self.cuda()
        return self

    def save(self,fname=None):
        import time
        self.cpu()
        if not fname:
            fname = 'data.{}.ds'.format(int(time.time()))
        torch.save({
            'inputs': self._data['inputs'],
            'targets': self._data['targets']
        },fname)

    def __getitem__(self,idx):
        """
            By default only return the training set, but this canbe toggled with calling either '.eval()' or '.train()' methods 
        """
        # assert idx < self.size
        return (self._data['inputs'][ self.idxs[idx] ],self._data['targets'][ self.idxs[idx] ].float())


