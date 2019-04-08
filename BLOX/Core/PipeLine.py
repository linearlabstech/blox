
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


class DummyClass:

    def __init__(self):
        pass

    def __call__(self,x):
        return x

class ParallelPipe:
    models = {}
    def __init__(self,cfg):
        cfg = sod(cfg)
        for k,v in cfg.items():
            self.models[k] = Compile( json.loads(open(cfg[k],'r').read() ) )
    def load(self,files):
        for k,v in files.items():
            if k in self.models: self.models[k].load_state_dict(torch.load(v))

    def __call__(self,x):
        return dict(
            zip(
                self.models.keys(),
                [f(x) for f in self.models.values()]
            )
        )
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
    models={}
    def __init__(self,cfg):
        cfg = sod(cfg)
        self.cfg = cfg
        self.order = []
        for k,v in cfg['Nets'].items():
            self.order.append(k)
            self.models[k] = Compile( json.loads(open(cfg['Nets'][k],'r').read() ) ).eval() if isinstance(cfg['Nets'][k],str) else ParallelPipe( cfg['Nets'][k] )
        if "Load" in cfg:
            for k,v in cfg['Load'].items():
                try:
                    self.models[k].load_state_dict(torch.load(v))
                except:
                    self.models[k].load(cfg['Load'])

    def __call__(self,x):
        with torch.no_grad():
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

class PipeLine:

    pipes = []
    threads = []
    running = False
    backend = 'rmq'
    def __init__(self,cfg):
        self.cfg = sod(cfg)

    def __exit__(self):
        self.close()

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
        assert isinstance(pipe,Pipe), 'the pipe you want to add to the PipeLine is not a Pipe!'
        self.pipes.append(pipe)

