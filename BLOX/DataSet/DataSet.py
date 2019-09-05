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
from BLOX.DataSet.TestSet import TestSet
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
    is_categorical = False
    def __init__(self,data,eval_split=.15,dtype='float'):
        data = data if isinstance(data,dict) else torch.load(data)
        size = len(data['inputs'])
        self.type = dtype
        eval_size = int(float(size)*eval_split)
        self.tsize = size-eval_size
        self.dsize = eval_size
        self.tidxs = list(range(self.tsize))
        self.didxs = list(range(self.dsize))
        self._train = TestSet({
            'inputs':data['inputs'][eval_size:],
            'targets':data['targets'][eval_size:]
        })
        self._eval = TestSet({
            'inputs':data['inputs'][:eval_size],
            'targets':data['targets'][:eval_size]
        })
        self.total_size = size
        self._size = self.tsize
        self.idx = 0
        self.bsz = 1
        # if torch.cuda.is_available():self.cuda()

    def __len__(self):return self.n
    
    @property
    def n(self):
        return self.tsize if self.training else self.dsize

    def shuffle(self):
        shuffle(self.tidxs)
        shuffle(self.didxs)
        return self

    # def __iter__ (self):
    #     return self#iter( ((x,y) for (x,y) in self) )

    # def __next__ (self):
    #     try:
    #         (x,y) = self._train.__next__() if self.training else self._eval.__next__()
    #     except IndexError:
    #         print('error')
    #     self.idx += 1
    #     return x,y
    
    def batchify(self,bsz=64):
        self.bsz = bsz
        self.tsize = self.tsize // bsz
        self.dsize = self.dsize // bsz
        self._size = self.tsize if self.training else self.dsize
        self._train.batchify(bsz)
        self._eval.batchify(bsz)
        self.tidxs = list(range( self.tsize ))
        self.didxs = list(range( self.dsize ))
        return self

    @property
    def x(self):
        return self._train.x if self.training else self._eval.x
    @property
    def y(self):
        return self._train.y if self.training else self._eval.y

    def cuda(self):
        """
            Move data GPU
        """
        self._train.cuda()
        self._eval.cuda()

    def size(self):
        return self._train.size() if self.training else self._eval.size()

    def cpu(self):
        """
            Move data CPU
        """
        self._train.cpu()
        self._eval.cpu()

    def eval(self):
        """
            switch to the evalelopment data
        """
        self.training = False
        self._size = self.dsize
        return self

    def pad(self):
        self._train.pad()
        self._eval.pad()
        return self

    def to(self, dtype):
        dtype = dtype.lower()
        assert dtype in ['cpu','gpu'], 'evalice type not supported'
        self.cpu() if dtype == 'cpu' else self.gpu()
        return self

    def categorical(self):
        if not self.is_categorical:
            self._train.categorical()
            self._eval.categorical()
            self.is_categorical = True
        return self

    # def float(self):
    #     self._data['inputs'].float()
    #     self._data['targets'].float()
    #     return self

    # def long(self):
    #     self._data['inputs'].long()
    #     self._data['targets'].long()
    #     return self

    def train(self):
        """
            switch to the training data
        """
        self.training = True
        self._size = self.tsize
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
        assert idx <= self.n
        if self.training: return self._train[ self.tidxs[idx] ]
        else: return self._eval[ self.didxs[idx] ]


