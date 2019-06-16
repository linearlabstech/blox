
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
import argparse,json,torch
from yahoo_finance_api2 import share
from Robinhood import Robinhood
# import time
# import torch.tensor as tt
# from BLOX.Common.Compiler import Compile,cfg2nets
from BLOX.Common.utils import ClearLine
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def update_balance(pv,cv): pass
import random
import datetime
def pg(v):

    print( 'Current account value: ' + bcolors.BOLD +  bcolors.OKGREEN + '${:.6}'.format(v) + bcolors.ENDC )

def pb(v):

    print( 'Current account value: ' + bcolors.BOLD +  bcolors.FAIL + '${:.6}'.format(v) + bcolors.ENDC  )
def pn(v):

    print( 'Current account value: ' + bcolors.BOLD +  bcolors.WARNING + '${:.6}'.format(v) + bcolors.ENDC  )

def print_summary(shares):
    # ClearLine()
    # ClearLine()
    # ClearLine()
    print('Current Time: {}'.format(datetime.datetime.now()) )
    print('Holding {} shares of stock'.format(shares))

    
DEFAULT_SYMBOL = 'MSFT'
my_share = share.Share('MSFT')
VALUE = 10000.
rows = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                1,
                                share.FREQUENCY_TYPE_MINUTE,
                                1)
cost = rows['high'][-1]
WALLET = VALUE - (cost * 5)
INITIAL_VALUE = VALUE
LAST_SALE = WALLET
SHARES = 5
LAST_VALUE = VALUE
import time
while True:
    action = random.randint(0,21) - 8
    rows = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                    1,
                                    share.FREQUENCY_TYPE_MINUTE,
                                    1)
    # action = -SHARES if action < 0 and -action < SHARES else action
    cost = rows['high'][-1]
    WALLET -= cost*action
    
    SHARES += action
    # if action < 0:
    #     # val = cost*action
    #     LAST_SALE = WALLET
    #     diff =  LAST_SALE - VALUE
    #     VALUE += diff
    # added_value = 
    VALUE = (cost*SHARES)+WALLET
    print(cost,SHARES,WALLET)
    # print_summary(SHARES)
    # if VALUE > INITIAL_VALUE:pg(VALUE)
    # elif VALUE < INITIAL_VALUE:pb(VALUE)
    # else:pn(VALUE)
    time.sleep(5)
    LAST_VALUE = VALUE


 


