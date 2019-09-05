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
import torch.nn.utils.rnn as rnn_utils
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
    _size = (0,)
    _data = None
    is_categorical = False
    def __init__(self,data,eval_split=.15,dtype='float'):
        self._data = data if isinstance(data,dict) else torch.load(data)
        if not isinstance(self._data['inputs'],torch.Tensor):
            self._data['inputs'] = [t.view(-1) for t in self._data['inputs']]
            self._data['targets'] = [t.view(-1) for t in self._data['targets']]
            self.pad()
        size = len(self._data['inputs'])
        self.type = dtype
        self.n = size
        self.idxs = list(range(self.n))
        # if torch.cuda.is_available():self.cuda()
        self.idx = 0
        self.bsz = 1
        self._size = self._data['targets'].size()

    def size(self):
        return self._size

    def __len__(self):return self.n
    
    def shuffle(self):
        shuffle(self.idxs)
        return self

    def reset(self):
        self.idx = 0
        return self

    @property
    def x(self):
        return self._data['inputs']
    @property
    def y(self):
        return self._data['targets']

    def batchify(self,bsz=64):
        self.bsz = bsz
        self.n = self.n // bsz
        nbatch = self._data['targets'].size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self._data['targets'].narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(-1,bsz).contiguous()
        self._data['targets'] = data
        data = self._data['inputs'].narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        s = self._data['inputs'].size()
        data = data.view(-1,bsz,s[-1]).contiguous()
        self._data['inputs'] = data
        self.idxs = list(range(self.n))

    def categorical(self):
        if not self.is_categorical:
            if self._data['targets'][0].size()[-1] > 1 or isinstance(self._data['targets'][0], (torch.FloatTensor,)): self._data['targets'] = [ torch.tensor([t.argmax()]).to( 'cuda' if torch.cuda.is_available() else 'cpu' ) for t in self._data['targets'] ]
        self.is_categorical = True
        self._data['targets'] = rnn_utils.pad_sequence(self._data['targets'], batch_first=True).clone()
        self._data['targets'].long()

        return self
    
    def pad(self):
        
        self._data['inputs'] = rnn_utils.pad_sequence(self._data['inputs'], batch_first=True)
        self._data['targets'] = rnn_utils.pad_sequence(self._data['targets'], batch_first=True)
        return self

    # def __iter__(self):
    #     return self

    # def __next__(self):
    #     try:
    #         x,y = self._data['inputs'][self.idx],self._data['targets'][self.idx]
    #     except IndexError:
    #         self.reset()
    #         return self.__next__()
    #     self.idx += 1
    #     return x,y

    # def float(self):
    #     self._data['inputs'].float()
    #     self._data['targets'].float()
    #     return self

    def long(self):
        self._data['inputs'] = self._data['inputs'].long()
        self._data['targets'] = self._data['targets'].long()
        return self
    
    @property
    def x(self):
        return self._data['inputs']

    @property
    def y(self):
        return self._data['targets']
    
    def cuda(self):
        """
            Move data GPU
        """
        if not isinstance(self._data['inputs'],torch.Tensor):
            for i in range(self.n):
                self._data['inputs'][i] = self._data['inputs'][i].cuda()
                self._data['targets'][i] = self._data['targets'][i].cuda()
        else:
            self._data['inputs'] = self._data['inputs'].cuda()
            self._data['targets'] = self._data['targets'].cuda()

    def cpu(self):
        """
            Move data CPU
        """
        if not isinstance(self._data['inputs'],torch.Tensor):
            for i in range(self.tsize):
                self._data['inputs'][i] = self._data['inputs'][i].cpu()
                self._data['targets'][i] = self._data['targets'][i].cpu()
        else:
            self._data['inputs'] = self._data['inputs'].cuda()
            self._data['targets'] = self._data['targets'].cuda()


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
        return (self._data['inputs'][ self.idxs[idx] ],self._data['targets'][ self.idxs[idx] ])


