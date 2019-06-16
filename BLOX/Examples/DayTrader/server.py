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
import time
import torch.tensor as tt
from BLOX.Common.Compiler import Compile,cfg2nets
def predict(nets,order,inp):
    for net in order:
        try:
            if isinstance(inp,tuple):(inp,h) = inp
            inp = nets[net](inp,h)
        except:
            # print(inp);print(nets[net](inp))
            if isinstance(inp,tuple):(inp,h) = inp
            
            inp = nets[net](inp)
    if isinstance(inp,tuple):(inp,h) = inp
    return inp.cpu().data.numpy()


DEFAULT_SYMBOL = 'MSFT'
my_share = share.Share('MSFT')

config = json.load(open('config.json','r'))
order, nets = cfg2nets(config)
nets['DayTrader'].load_state_dict(torch.load('DayTrader.pt'))
while True:
    rows = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                        1,
                                        share.FREQUENCY_TYPE_MINUTE,
                                        1)
    inp = tt([ [ [ rows['open'][-1],rows['close'][-1],rows['volume'][-1],rows['low'][-1],rows['high'][-1] ] ] ])
    y = predict(nets,order,inp)[0]
    action = decide( y )
    print(action, '@',rows['high'][-1],y)
    time.sleep(60)

# while True:



# parser = argparse.ArgumentParser()
# parser.add_argument('-u','--username',help='your Robinhood username',required=True)
# parser.add_argument('-p','--password',help='your Robinhood password',required=True)
# parser.add_argument('-t','--ticker',help='the ticker symbol you wish to monitor',default=DEFAULT_SYMBOL)
# args = parser.parse_args()
# #Setup
# my_trader = Robinhood()
# #login
# my_trader.login(username=args.username, password=args.password)

# #Get stock information
#     #Note: Sometimes more than one instrument may be returned for a given stock symbol
# stock_instrument = my_trader.instruments("GEVO")[0]

# #Get a stock's quote
# my_trader.print_quote("AAPL")

#Prompt for a symbol
# my_trader.print_quote()

# #Print multiple symbols
# my_trader.print_quotes(stocks=["BBRY", "FB", "MSFT"])