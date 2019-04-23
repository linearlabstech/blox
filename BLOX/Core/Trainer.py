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
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import yajl as json
from tensorboardX import SummaryWriter
# from .build_vocab import Vocabulary
# from .model import EncoderCNN, DecoderRNN

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from ..Common.utils import *
from ..Common.Compiler import Compile,cfg2nets
from ..DataSet.DataSet import DataSet
# 
TEXT = \
"""

 .----------------. .----------------. .----------------. .----------------. 
| .--------------. | .--------------. | .--------------. | .--------------. |
| |   ______     | | |   _____      | | |     ____     | | |  ____  ____  | |
| |  |_   _ \    | | |  |_   _|     | | |   .'    `.   | | | |_  _||_  _| | |
| |    | |_) |   | | |    | |       | | |  /  .--.  \  | | |   \ \  / /   | |
| |    |  __'.   | | |    | |   _   | | |  | |    | |  | | |    > `' <    | |
| |   _| |__) |  | | |   _| |__/ |  | | |  \  `--'  /  | | |  _/ /'`\ \_  | |
| |  |_______/   | | |  |________|  | | |   `.____.'   | | | |____||____| | |
| |              | | |              | | |              | | |              | |
| '--------------' | '--------------' | '--------------' | '--------------' |
 '----------------' '----------------' '----------------' '----------------' 

"""
# TEXT = \
# """
# __/\\\\\\\\\\\\\____/\\\__________________/\\\\\_______/\\\_______/\\\_        
#  _\/\\\/////////\\\_\/\\\________________/\\\///\\\____\///\\\___/\\\/__       
#   _\/\\\_______\/\\\_\/\\\______________/\\\/__\///\\\____\///\\\\\\/____      
#    _\/\\\\\\\\\\\\\\__\/\\\_____________/\\\______\//\\\_____\//\\\\______     
#     _\/\\\/////////\\\_\/\\\____________\/\\\_______\/\\\______\/\\\\______    
#      _\/\\\_______\/\\\_\/\\\____________\//\\\______/\\\_______/\\\\\\_____   
#       _\/\\\_______\/\\\_\/\\\_____________\///\\\__/\\\_______/\\\////\\\___  
#        _\/\\\\\\\\\\\\\/__\/\\\\\\\\\\\\\\\___\///\\\\\/______/\\\/___\///\\\_ 
#         _\/////////////____\///////////////______\/////_______\///_______\///__


# """
writer = SummaryWriter()
SCALARS = {
}

def register_scalar(obj,key):
    SCALARS[key] = writer.add_scalar


class Trainer:

    def __init__(self,args):
        try: self.config = json.load(open(args.config,'r'))
        except: 
            if isinstance(args,dict): self.config = args
            elif isinstance(args,str): self.config = json.load(open(args,'r'))
            else:raise ValueError('Incorrect data type passed to Trainer class')

        if self.config['Verbose']:print(TEXT+('\n'*8))
        # for _ in range(13):ClearLine()
    
    def build(self):pass

    def run(self):
        config = self.config
        order, nets = cfg2nets(config)
        opt = GetOptim([ p for m in config['Optimizer']['Params'] for p in nets[m].parameters() ],config['Optimizer']['Algo'],config['Optimizer']['Kwargs'] )
        loss = GetLoss(config['Loss']['Algo'],config['Loss']['Kwargs'])
        writer = SummaryWriter(config['TensorboardX']['Dir']) if 'Dir' in config['TensorboardX'] else None
        if writer:
            register_scalar(loss,'Loss')
            register_scalar(get_acc,'Acc')
        data_set = DataSet(config['DataSet'])
        
        i,t = data_set[9]
        if writer and config['TensorboardX']['SaveGraphs']:
            for net in order:
                with SummaryWriter(comment=f' {net}') as w:w.add_graph(nets[net], i)
            i = nets[net]( i )
        tlosses = np.zeros(config['Epochs'])
        dlosses = np.zeros(config['Epochs'])
        for e in range(config['Epochs']):
            acc = 0
            if config['Verbose']:print('[{}] Epoch training..'.format(e+1))
            data_set.train()
            ttloss = 0
            tdloss = 0
            for idx,(inp,targ) in enumerate(tqdm.tqdm(data_set) if config['Verbose'] else data_set) :
                opt.zero_grad()
                if inp.shape[1] < 3: continue
                # try:
                for net in order:inp = nets[net](inp)
                l = loss(inp,targ)
                ttloss += l
                l.backward()
                opt.step()
                acc += (1. if inp.argmax() == targ.argmax() else 0.) / float(idx+1)
                if (idx%(config['TensorboardX']['LogEvery']+1))==0 and config['TensorboardX']['LogEvery'] > 0 and writer:
                    for key in config['TensorboardX']['Log']:
                        if key in SCALARS:SCALARS[key]('{} {}'.format('train:' if data_set.training else 'dev:',  key),l.item() if key == 'Loss' else acc ,(e+1)*data_set.size )
                if (idx%(config['SaveEvery']+1))==0 and config['SaveEvery'] > 0:
                    for m,net in nets.items():
                        torch.save(net.state_dict(),'{}-{}'.format(m,config['FileExt']) )
            data_set.dev()
            with torch.no_grad():
                for idx,(inp,targ) in enumerate(tqdm.tqdm(data_set) if config['Verbose'] else data_set) :
                    try:
                        for net in nets.values():inp = net(inp)
                        if (idx%(config['TensorboardX']['LogEvery']+1))==0 and config['TensorboardX']['LogEvery'] > 0 and writer:
                            for key in config['TensorboardX']['Log']:
                                if key in SCALARS:SCALARS[key]('{} {}'.format('train:' if data_set.training else 'dev:',  key),l.item(),(e+1)*data_set.size )
                    except:continue
                    l = loss(inp,targ)
                    tdloss+= l
            
            if config['Verbose']:PrintSummary(tlosses,ttloss,dlosses,tdloss,e,config['Epochs'])
            
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LL Training Application')
    parser.add_argument('-c','--config', default='config.json')
    args = parser.parse_args()
    main(args)