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
import argparse
import torch
import tqdm
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import yajl as json
from tensorboardX import SummaryWriter
import random
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from ..Common.utils import *
from ..Common.Compiler import Compile,cfg2nets
from ..DataSet.TestSet import TestSet
from BLOX.Modules.ReplayMemory import ReplayMemory
from collections import namedtuple
from itertools import count

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from BLOX.Common.Strings import TITLE as TEXT
from BLOX.Core.Recordable import METRICS,ADDITIONAL_METRICS,Vizzy

class Tester:

    def __init__(self,args):
        try: self.config = json.load(open(args.config,'r'))
        except: 
            if isinstance(args,dict): self.config = args
            elif isinstance(args,str): self.config = json.load(open(args,'r'))
            else:raise ValueError('Incorrect data type passed to Trainer class')

        if self.config['Verbose']:print(TEXT+('\n'*8))

    def run(self):
        config = self.config
        torch.cuda.empty_cache()

        model = cfg2nets(config)

        # data = DataLoader(TestSet(config['DataSet']), batch_size=config['BatchSize'] if 'BatchSize' in config else 1,shuffle=True )
        data = TestSet(config['DataSet'])
        writer = SummaryWriter(config['TensorboardX']['Dir'] if 'Dir' in config['TensorboardX'] else 'runs')

        tlosses = np.zeros(config['Epochs'])
        dlosses = np.zeros(config['Epochs'])
        evaluator = create_supervised_evaluator(model,
                                                device=device)

        for m in config['TensorboardX']['Log']:
            if m not in METRICS or m == 'Loss':continue
            mtrc = METRICS[m]()
            mtrc.attach(evaluator,m)


        pbar = tqdm.tqdm(
            initial=0, leave=False, total=len(data),
        )

        add_metrics = {}
        @evaluator.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            i = engine.state.iteration 
            if (i%(config['TensorboardX']['LogEvery']))==0 and config['TensorboardX']['LogEvery'] > 0 and writer:
                for m in engine.state.metrics.keys():
                    if m in METRICS:writer.add_scalar(m, engine.state.metrics[m], engine.state.iteration)
                try:
                    for m in config['TensorboardX']['Log']:

                        if m in ADDITIONAL_METRICS:
                            if m not in add_metrics:
                                add_metrics[m] = {
                                    'y_h':[],
                                    'y':[]
                                }
                            add_metrics[m]['y'].append( engine.state.output[1].view(-1).numpy() )
                            add_metrics[m]['y_h'].append( engine.state.output[0].view(-1).data.numpy() )
                except:pass

                pbar.update(config['TensorboardX']['LogEvery'])
        try:
            evaluator.run(data, max_epochs=1)
        except:pass
        pbar.close()
        try:
            for m in config['TensorboardX']['Log']:
                if m in ADDITIONAL_METRICS:
                    getattr(Vizzy,m)(Vizzy,ADDITIONAL_METRICS[m]( add_metrics[m]['y_h'],add_metrics[m]['y'] ))
        except Exception as e:pass
        
