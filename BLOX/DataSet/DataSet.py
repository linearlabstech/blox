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
    _eval = None
    training = True
    def __init__(self,data,eval_split=.15,dtype='float'):
        data = data if isinstance(data,dict) else torch.load(data)
        size = len(data['inputs'])
        self.type = dtype
        eval_size = int(float(size)*eval_split)
        self.tsize = size-eval_size
        self.dsize = eval_size
        self.tidxs = list(range(self.tsize))
        self.didxs = list(range(self.dsize))
        self._train = {
            'inputs':data['inputs'][eval_size:],
            'targets':data['targets'][eval_size:]
        }
        self._eval = {
            'inputs':data['inputs'][:eval_size],
            'targets':data['targets'][:eval_size]
        }
        self.total_size = size
        self.size = self.tsize
        if torch.cuda.is_available():self.cuda()

    def __len__(self):return self.size
    
    def shuffle(self):
        shuffle(self.tidxs)
        shuffle(self.didxs)
        return self
    
    def cuda(self):
        """
            Move data GPU
        """
        for i in range(self.tsize):
            self._train['inputs'][i] = self._train['inputs'][i].cuda()
            self._train['targets'][i] = self._train['targets'][i].cuda()
        for i in range(self.dsize):
            self._eval['inputs'][i] = self._eval['inputs'][i].cuda()
            self._eval['targets'][i] = self._eval['targets'][i].cuda()

    def cpu(self):
        """
            Move data CPU
        """
        for i in range(self.tsize):
            self._train['inputs'][i] = self._train['inputs'][i].cpu()
            self._train['targets'][i] = self._train['targets'][i].cpu()
        for i in range(self.dsize):
            self._eval['inputs'][i] = self._eval['inputs'][i].cpu()
            self._eval['targets'][i] = self._eval['targets'][i].cpu()

    def eval(self):
        """
            switch to the evalelopment data
        """
        self.training = False
        self.size = self.dsize
        return self

    def to(self, dtype):
        dtype = dtype.lower()
        assert dtype in ['cpu','gpu'], 'evalice type not supported'
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
            'inputs': self._train['inputs']+self._eval['inputs'],
            'targets': self._train['targets']+self._eval['targets']
        },fname)

    def __getitem__(self,idx):
        """
            By default only return the training set, but this canbe toggled with calling either '.eval()' or '.train()' methods 
        """
        # assert idx < self.size
        if self.training: return (self._train['inputs'][ self.tidxs[idx] ],self._train['targets'][ self.tidxs[idx] ].float())
        else: return (self._eval['inputs'][ self.didxs[idx] ],self._eval['targets'][self.didxs[idx]].float())


