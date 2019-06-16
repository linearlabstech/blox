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
import argparse
from yahoo_finance_api2 import share
import torch.tensor as tt
import torch,random

def get_targets(rows):
    """
        This is where you can build your target.
        For the tutorial we're only concerned about whether we should buy or sell.
        So we'll create a rule for this, we'll need to set this up as regression ( on the open set (-1,1) ). Our rule is as follows:
        If the stock closes below the open, we should have sold (t-1 = sell at close, t = buy at close).
        If the stock closes above the open, we should have bought (t-1 = buy at close, t = sell at close).
        While under the constraint of maximizing our profit

        We can obviously make this more complex, even with the data we have, but for this tutorial, 


        If you want to create your own targets, this is where you should do it. Below is the accessible data structure passed to this function.

            rows = {
                    "timestamp": [
                        1557149400000,
                    ],
                    "open": [
                        126.38999938964844,
                    ],
                    "high": [
                        128.55999755859375,
                    ],
                    "low": [
                        126.11000061035156,
                    ],
                    "close": [
                        128.14999389648438,
                    ],
                    "volume": [
                        24239800,
                    ]
                }
    """

    

    # targets: sell = -1, buy = +1
    # set to sell at beginning of the trading day
    # we assume that unless the it's going down, buy.
    # later we'll add some business logic to determine the actual action of purchasing
    # return [ tt([0.]) ] + [ tt([ 0 if (rows['close'][i-2] > rows['open'][i-2]) and (rows['close'][i] >  rows['open'][i]) else (1 if random.random() > .7 else 2 )]) for i in range(2,len(rows['open'])) ]

    return [ tt( [ [ [ rows['high'][i] ] ] ] )  for i in range(1,len(rows['open'])) ]

def get_inputs(rows):
    # you could also use a pandas DataFrame
    return [ tt( [ [ [ rows['open'][i],rows['close'][i],rows['volume'][i],rows['low'][i],rows['high'][i] ] ] ]) for i in range(len(rows['open'])-1 ) ]

def main(args):
    # default grab the last 75 days
    import datetime
    if args.csv:
        import pandas as pd
        data = pd.read_csv(args.csv)
    else:
        today = datetime.date.today()
        ticker = share.Share(args.ticker)
        data = ticker.get_historical(share.PERIOD_TYPE_DAY,args.start,share.FREQUENCY_TYPE_MINUTE,int(60/args.frequency))
    torch.save({
        'inputs':get_inputs(data),
        'targets':get_targets(data)
    },args.output_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--ticker',help="enter the stock ticker symbol",required=True)
    parser.add_argument('-s','--start',help="start date of data to grab. default is 75 days ago",default=75,type=int)
    parser.add_argument('-o','--output_file',help="name of the output file to save the dataset",default='trader.ds')
    parser.add_argument('-f','--frequency',help='how frequent to sample each day of trading (in hourly fractions)',type=int,default=1)
    parser.add_argument('--csv',help='the csv file to load instead of downloading fresh data',default=None)

    main( parser.parse_args() )