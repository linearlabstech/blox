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
import yajl as json,os, torch
import xml.etree.ElementTree as ET
from torchvision import transforms

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def ndarray2pt(array):
    return [ array2pt(t) if isinstance(t[0],(int,float)) else ndarray2pt(t) for t in array]


def array2pt(array):
    return [ torch.tensor(t) for t in array]

def df2ds(self,df,inputs,target):
    """
        Consume Pandas Dataframe object with the inputs and outputs.
    """
    cats = set([object])
    in_tables = {}
    table = None
    inputs = inputs if isinstance(inputs,list) else [inputs]
    for i in inputs:
        if df[i].dtype in cats:
            unique = df[target].unique()
            in_tables[i] = dict(zip(unique,range(len(unique))))
    if df[target].dtype in cats:
        unique = df[target].unique()
        table = dict(zip(unique,range(len(unique))))
    return DataSet({
            'inputs': [ array2pt( [ df[i][r] if not in in_tables else in_tables[i][df[i][r]] for i in inputs ] ) for r in range(df.shape[0]) ] ,
            'targets':[ array2pt([table[v]]) if table else [v]  for v in df[target] ]
    })

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
def img2tensor(image, transform_img=True):
    if isinstance(image,torch.Tensor):return image
    image = Image.open(image)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform_img:
        image = transform(image).unsqueeze(0)

    return image

