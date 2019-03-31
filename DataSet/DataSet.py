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

class DataSet:
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
    _train = None
    _dev = None
    training = True
    def __init__(self,data,dev_split=.15,dtype='float'):
        data = data if isinstance(data,dict) else torch.load(data)
        size = len(data['inputs'])
        self.type = dtype
        dev_size = int(float(size)*dev_split)
        self.tsize = size-dev_size
        self.dsize = dev_size
        self._train = {
            'inputs':data['inputs'][dev_size:],
            'targets':data['targets'][dev_size:]
        }
        self._dev = {
            'inputs':data['inputs'][:dev_size],
            'targets':data['targets'][:dev_size]
        }
        self.total_size = size
        self.size = self.tsize

    def __len__(self):return self.size
    
    def shuffle(self):
        shuffle(self._train['inputs'])
        shuffle(self._train['targets'])
        shuffle(self._dev['inputs'])
        shuffle(self._dev['targets'])
        return self
    
    def cuda(self):
        """
            Move data GPU
        """
        for i in range(self.tsize):
            self._train['inputs'][i].cuda()
            self._train['targets'][i].cuda()
        for i in range(self.dsize):
            self._dev['inputs'][i].cuda()
            self._dev['targets'][i].cuda()

    def cpu(self):
        """
            Move data CPU
        """
        for i in range(self.tsize):
            self._train['inputs'][i].cpu()
            self._train['targets'][i].cpu()
        for i in range(self.dsize):
            self._dev['inputs'][i].cpu()
            self._dev['targets'][i].cpu()

    def dev(self):
        """
            switch to the development data
        """
        self.training = False
        self.size = self.dsize
        return self

    def to(self, dtype):
        dtype = stype.lower()
        assert dtype in ['cpu','gpu'], 'Device type not supported'
        self.cpu() if dtype == 'cpu' else self.gpu()
        return self

    def train(self):
        """
            switch to the training data
        """
        self.training = True
        self.size = self.tsize
        return self

    def save(self,fname=None):
        import time
        self.cpu()
        if not fname:
            fname = 'data.{}.ds'.format(int(time.time()))
        torch.save({
            'inputs': self._train['inputs']+self._dev['inputs'],
            'targets': self._train['targets']+self._dev['targets']
        },fname)

    def __getitem__(self,idx):
        """
            By default only return the training set, but this canbe toggled with calling either '.dev()' or '.train()' methods 
        """
        # assert idx < self.size
        if self.training: return (self._train['inputs'][idx],self._train['targets'][idx].float())
        else: return (self._dev['inputs'][idx],self._dev['targets'][idx].float())


