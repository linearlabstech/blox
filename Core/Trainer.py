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

from JSON2Net import Compile




class Trainer:

    def __init__(self,args):
        try: self.config = json.load(open(args.config,'r'))
        except: 
            if isinstance(args,dict) self.config = args
            else:raise ValueError('Incorrect data type passed to Trainer class')
    
    def build(self):pass

    def run(self):
        config = self.config
        nets = {}
        for k,v in config['Nets'].items():nets[k] = Compile( json.load(open(config['Nets'][k],'r')) )
        opt = GetOptim([ p for m in config['Optimizer']['Params'] for p in nets[m].parameters() ],config['Optimizer']['Algo'],config['Optimizer']['Kwargs'] )
        loss = GetLoss(config['Loss']['Algo'],config['Loss']['Kwargs'])
        data_set = DataSet(config['DataSet'])
        i,t = data_set[9]
        for net,model in nets.items():
            with SummaryWriter(comment=f' {net}') as w:
                w.add_graph(model, i)
            i = model( i )
        tlosses = np.zeros(config['Epochs'])
        dlosses = np.zeros(config['Epochs'])
        for e in range(config['Epochs']):
            print('[{}] Epoch training..'.format(e+1))
            data_set.train()
            ttloss = 0
            tdloss = 0
            for idx,(inp,targ) in enumerate(tqdm.tqdm(data_set) if config['Verbose'] else data_set) :
                opt.zero_grad()
                if inp.shape[1] < 3: continue
                # try:
                for net in nets.values():inp = net(inp)
                l = loss(inp,targ)
                ttloss += l
                l.backward()
                opt.step()
                if (idx%(config['SaveEvery']+1))==0 and config['SaveEvery'] > 0:
                    for m,net in nets.items():
                        torch.save(net.state_dict(),'{}-{}'.format(m,config['FileExt']) )
            data_set.dev()
            with torch.no_grad():
                for idx,(inp,targ) in enumerate(tqdm.tqdm(data_set) if config['Verbose'] else data_set) :
                    try:
                        for net in nets.values():inp = net(inp)
                    except:continue
                    l = loss(inp,targ)
                    tdloss+= l
            
            PrintSummary(tlosses,ttloss,dlosses,tdloss,e,config['Epochs'])
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LL Training Application')
    parser.add_argument('-c','--config', default='config.json')
    args = parser.parse_args()
    main(args)