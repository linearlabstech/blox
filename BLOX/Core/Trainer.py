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
from ..DataSet.DataSet import DataSet
from BLOX.Modules.ReplayMemory import ReplayMemory
from BLOX.Common.Strings import TITLE as TEXT
from collections import namedtuple
from itertools import count

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


from BLOX.Core.Recordable import METRICS,ADDITIONAL_METRICS,Vizzy

def get_metric(m,loss,n_classes):
    # from BLOX.Core.Recordable import METRICS
    if m == "Loss":return METRICS[m](loss)
    elif m =='ConfusionMatrix':return METRICS[m](n_classes)
    else: return METRICS[m]()


class Trainer:

    def __init__(self,args):
        try: self.config = json.load(open(args.config,'r'))
        except: 
            if isinstance(args,dict): self.config = args
            elif isinstance(args,str): self.config = json.load(open(args,'r'))
            else:raise ValueError('Incorrect data type passed to Trainer class')

        if self.config['Verbose']:print(TEXT+('\n'*8))
  
    def update(self,opt,order,nets,config):
        if 'ClipGradient' in config:
            if config['ClipGradient'] > 0:
                for o in order:torch.nn.utils.clip_grad_norm_(nets[o].parameters(), config['ClipGradient'])
        opt.step()
    
    def run_qepoch(self,memory,opt,policy_net,target_net,loss,writer,env,e):
        config = self.config
        BATCH_SIZE = 2
        GAMMA = 0.999
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200
        def select_action(state,steps_done,policy_net):
            GAMMA = 0.999
            EPS_START = 0.9
            EPS_END = 0.05
            EPS_DECAY = 200
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    y = policy_net(state)
                    return (y.argmax() - (env.n_actions/2) ).view(1,1).long()
            else:
                return torch.tensor([[random.randrange( env.n_actions )]], dtype=torch.long)
        last_state = env.get_state()
        curr_state = env.get_state()
        state = curr_state - last_state
        for i in tqdm.tqdm(range(len(env)) if config['Verbose'] else range(len(env)-1)):
            
            action = select_action(state,i,policy_net)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward])
            if not done:
                next_state = env.get_state() #
            else:
                next_state = curr_state - last_state
                env.reset()

            memory.push(state, action, next_state, reward)

            state = next_state
            
            if len(memory) > BATCH_SIZE:

                transitions = memory.sample(BATCH_SIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                state_action_values = (policy_net(state_batch) )#.gather(0, action_batch) 
                next_state_values = torch.zeros(BATCH_SIZE)
                next_state_values[non_final_mask] = (target_net(non_final_next_states) ).detach().argmax(2).view(-1).float()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                
                l = loss(state_action_values, expected_state_action_values.unsqueeze(1))
                opt.zero_grad()
                l.backward()
                if (i%(config['TensorboardX']['LogEvery']+1))==0 and config['TensorboardX']['LogEvery'] > 0 and writer:
                    for key in config['TensorboardX']['Log']:
                        if hasattr(env,key):
                            SCALARS[key]('{}'.format(key),float(getattr(env,key)) ,i+(e*len(env)) )
                opt.step()


    def train_qlearners(self):

        config = self.config
        torch.cuda.empty_cache()
        policy_net = cfg2nets(config)
        loss = GetLoss(config['Loss']['Algo'],config['Loss']['Kwargs'])
        env = GetAction(config['Environment'])
        writer = SummaryWriter(config['TensorboardX']['Dir']) if 'Dir' in config['TensorboardX'] else None
        if writer:
            register_scalar(loss,'Loss')
            register_scalar(None,'Acc')
            for w in config['TensorboardX']['Log']:
                register_scalar(None,w)
        

        target_net = policy_net
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        env.n_actions =  policy_net[-2][-1].out_features
        opt =  GetOptim( policy_net.parameters(),config['Optimizer']['Algo'].replace('DQN',''),config['Optimizer']['Kwargs'] )

        memory = ReplayMemory(len(env))

        for e in range(config['Epochs']):
            self.run_qepoch(memory,opt,policy_net,target_net,loss,writer,env,e)
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(),'{}{}'.format(order[0],config['FileExt']) )
            env.reset()
            memory.clear()
            ClearLine()
    
    def train_ganlearners():pass

    def run(self):
        if 'dqn' in self.config['Optimizer']['Algo'][:3].lower():
            self.train_qlearners()
            return
        elif 'gan' in self.config['Optimizer']['Algo'][:3].lower():
            self.train_ganlearners()
            return
        config = self.config
        torch.cuda.empty_cache()

        model = cfg2nets(config)
        opt = GetOptim([ p for m in config['Optimizer']['Params'] for p in model.nets[m].parameters() ],config['Optimizer']['Algo'],config['Optimizer']['Kwargs'] )
        loss = GetLoss(config['Loss']['Algo'],config['Loss']['Kwargs'])
        # data = DataLoader(DataSet(config['DataSet']), batch_size=config['BatchSize'] if 'BatchSize' in config else 1,shuffle=True )
        data = DataSet(config['DataSet'])
        writer = SummaryWriter(config['TensorboardX']['Dir'] if 'Dir' in config['TensorboardX'] else 'runs')

        tlosses = np.zeros(config['Epochs'])
        dlosses = np.zeros(config['Epochs'])

        trainer = create_supervised_trainer(model, opt, loss, device=device,output_transform=lambda x, y, y_pred, loss: (y_pred, y,))

        for m in config['TensorboardX']['Log']:
            if m not in METRICS:continue
            mtrc = get_metric(m,loss,16)
            mtrc.attach(trainer,m)

        # evaluator = create_supervised_evaluator(model,
        #                                         metrics=dict(zip( config['TensorboardX']['Log'], [ get_metric(m,loss,16) for m in config['TensorboardX']['Log']  if m in METRICS] )),
        #                                         device=device)
        pbar = tqdm.tqdm(
            initial=0, leave=False, total=len(data),
        )

        add_metrics = {}
        model(data[0][0])
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_metrics(engine):
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
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            # evaluator.run(val_loader)
            # pbar.refresh()
            pbar.n = pbar.last_print_n = 0
        try:
            trainer.run(data, max_epochs=config['Epochs'])
        except:pass
        pbar.close()
        try:
            for m in config['TensorboardX']['Log']:
                if m in ADDITIONAL_METRICS:
                    getattr(Vizzy,m)(Vizzy,ADDITIONAL_METRICS[m]( add_metrics[m]['y_h'],add_metrics[m]['y'] ))
        except Exception as e:pass
