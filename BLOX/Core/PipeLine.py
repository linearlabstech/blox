
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

import torch,yajl as json
from multiprocessing import Process
from ..Common.Compiler import Compile
from ..Common.utils import StrOrDict as sod
from ..RabbitMQ.Host import worker
import torch
from torch import nn
class DummyClass:

    def __init__(self):
        pass

    def __call__(self,x):
        return x
class ParallelPipe:
    def __init__(self,cfg,as_dict=True):
        # super(ParallelPipe,self).__init__()
        cfg = sod(cfg)
        self.models = {}
        self.as_dict=as_dict
        self.no_grad = False
        for k,v in cfg.items():
            self.models[k] = Compile( json.loads(open(cfg[k],'r').read() ) ) if isinstance(cfg[k],str) else ParallelPipe( cfg[k] )
    def load(self,files):
        for k,v in files.items():
            if k in self.models: self.models[k].load_state_dict(torch.load(v))

    def train(self):
        self.no_grad = False
        for f in self.models.keys():self.models[f].train()
    
    def eval(self):
        self.no_grad = True
        for f in self.models.keys():self.models[f].eval()

    def __call__(self,x):
        if self.no_grad:
            with toch.no_grad(): return dict(
                zip(
                    self.models.keys(),
                    [f(x) for f in self.models.values()]
                )
            ) if self.as_dict else \
            [f(x) for f in self.models.values()]
        else: return dict(
                zip(
                    self.models.keys(),
                    [f(x) for f in self.models.values()]
                )
            ) if self.as_dict else \
            [f(x) for f in self.models.values()]

    def __str__(self):
        import texttable as tt
        tab = tt.Texttable()
        headings = list(self.models.keys())
        tab.header(headings)
        tab.add_row(self.models.items())
        return tab.draw()
    
    def __repr__(self):
        return self.__str__()

class Pipe:
    def __init__(self,cfg):
        # super(Pipe,self).__init__()
        cfg = sod(cfg)
        self.cfg = cfg
        self.order = []
        self.models = {}
        self.no_grad = False
        for k,v in cfg.items():
            if "Load" in k:continue
            self.order.append(k)
            self.models[k] = Compile( json.loads(open(v,'r').read() ) ) if isinstance(v,str) else ParallelPipe( v )

        if "Load" in cfg:
            for k,v in cfg['Load'].items():
                try:
                    self.models[k].load_state_dict(torch.load(v))
                except:
                    self.models[k].load(cfg['Load'])

    def train(self):
        self.no_grad = False
        for f in self.order:self.models[f].train()
    
    def eval(self):
        self.no_grad = True
        for f in self.order:self.models[f].eval()
    

    def __call__(self,x):
        if self.no_grad:
            with torch.no_grad():
                for f in self.order: x = self.models[f](x)
        else:
            for f in self.order: x = self.models[f](x)
        return x

    
    def __str__(self):
        import texttable as tt
        s = ''
        for k in self.models.keys():
            tab = tt.Texttable()
            headings = [k]
            tab.header(headings)
            tab.add_row( [ self.models[k] ] )
            s += tab.draw() + '\n'
        return s

class PipeLine(nn.Module):
    backend = 'rmq'
    def __init__(self,cfg=None):
        super(PipeLine,self).__init__()
        self.pipes = []
        self.threads = []
        self.running = False
        if cfg:self.cfg = sod(cfg)

    def __exit__(self, exc_type, exc_value, traceback):self.close()

    def __str__(self):
        s = ''
        for p in self.pipes:
            s += str(p)+'\n'
        return s
    
    def __repr__(self):
        return self.__str__()

    def close(self):
        if self.running:
            for t in self.threads:
                t.join()
                t.close()
        self.running = False

    def __call__(self,x):
        for f in self.pipes:x = f(x)
        return x

    def run(self):
        self.running = True
        self.threads = []
        if self.backend == 'rmq':
            for i in range(self.cfg['threads']):
                self.threads.append(Process(target=worker,args=(self.cfg['addr'],self.cfg['queue'],self.pipes  )))
                self.threads[i].start()

    def register_pipes(self,pipes):
        for p in pipes:self.register_pipe(p)

    def register_pipe(self,pipe):
        assert isinstance(pipe,(Pipe,ParallelPipe)), 'the pipe you want to add to the PipeLine is not a Pipe!'
        self.pipes.append(pipe)
        return self

