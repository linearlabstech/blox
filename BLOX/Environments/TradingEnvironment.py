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

class TradingEnvironment(object):

    def __init__(self,fname,initial_budget=10000.0,initial_purchase=1,final_budget=100000):
        '''
            POSSIBLE STATES
            - SELL: (-inf,0] 
            - HOLD: (0)
            - BUY:  [0,inf)
        '''
        data = torch.load(fname)
        self.inputs = data['inputs']
        self.targets = data['targets']
        assert len(self.targets) == len(self.inputs)
        self.idx = -1
        self.final_budget = final_budget
        self.initial_purchase = initial_purchase
        self.initial_budget = initial_budget
        # how many shares we're starting off with
        self.value = self.targets[0]*initial_purchase
        # how much we've got left in the bank
        self.wallet = initial_budget-self.value
        self.n_shares = initial_purchase
        self.last_sold = self.value
        self.loss = 0
        self.gainz = 0

        # half this value is the max amount of shares per transaction 
        self.n_actions = 21

    def __len__(self):
        return len(self.targets)

    def get_state(self):
        self.idx+=1
        if self.idx >= len(self.inputs)-1:self.idx = 0
        return self.inputs[self.idx]

    def step(self,action):
        curr_price = self.targets[self.idx]

        pv = self.value

        # checking to see if it's out of bounds
        max_buy = int(self.wallet.float()/curr_price)

        oob_error = ( (max_buy - action) if action > max_buy else 0 ) if action > 0 else ( (self.n_shares + action) if -action > self.n_shares else 0  ) 

        action += oob_error   

        # this is where we determine the loss of the action taken
        gainz = 0.
        if action < 0 and self.n_shares < 0:
            action = 0
        if action < 0 and -action > self.n_shares:
            action = -self.n_shares
        self.last_sold = self.wallet
        self.n_shares += action
        self.wallet -= (action*curr_price)
        
        # we're selling the stock  
        if action < 0 and self.n_shares > 0:
            ganiz = self.wallet - self.last_sold            

        

        # if want to make it do this as fast as possible, we should add some error for how long it takes.
        # adding an exponential term would make it go much faster (like taking the log of the inverse fraction)
        # loss -= float( self.idx/len(self.targets) ) 
        self.value = gainz + (self.n_shares * curr_price) + self.wallet

        self.loss = self.value - pv


        return self.inputs, self.loss , self.idx == len(self.inputs) or self.wallet >= self.final_budget,{} 

    def reset(self):
        self.idx = -1
        self.value = self.initial_budget#self.targets[0]*self.initial_purchase
        self.wallet = self.initial_budget-(self.targets[0]*self.initial_purchase)
        self.n_shares = self.initial_purchase
        self.last_sold = self.value
        self.loss = 0
        self.gainz = 0

