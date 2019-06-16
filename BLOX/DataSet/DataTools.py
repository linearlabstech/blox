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
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from html.parser import HTMLParser
from .DataSet import DataSet
import gym
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

def cart_sr(screen,env):
    _, screen_height, screen_width = screen.shape
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    cart_location = int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    return slice_range
sr_table = {
    'CartPole-v0':cart_sr
}

# def Gym2DataSet(gym_env,size=10000,save=False):
#     env = gym.make('CartPole-v0').unwrapped
#     resize = transforms.Compose([transforms.ToPILImage(),
#                     transforms.Resize(40, interpolation=Image.CUBIC),
#                     transforms.ToTensor()])
#     def get_screen():
#         # Returned screen requested by gym is 400x600x3, but is sometimes larger
#         # such as 800x1200x3. Transpose it into torch order (CHW).
#         screen = env.render(mode='rgb_array').transpose((2, 0, 1))
#         # Cart is in the lower half, so strip off the top and bottom of the screen
#         _, screen_height, screen_width = screen.shape
#         screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
#         view_width = int(screen_width * 0.6)
#         slice_range = sr_table[gym_env](screen,env)

#         # Strip off the edges, so that we have a square image centered on a cart
#         screen = screen[:, :, slice_range]
#         # Convert to float, rescale, convert to torch tensor
#         # (this doesn't require a copy)
#         screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
#         screen = torch.from_numpy(screen)
#         # Resize, and add a batch dimension (BCHW)
#         return resize(screen).unsqueeze(0)
#     n_actions = env.action_space.n
#     env.reset()
#     inps = []
#     for i in range(size):
#         last_screen = get_screen()
#         current_screen = get_screen()
#         inps.append(current_screen - last_screen)

#     return DataSet({'inputs':inps,'targets':[0]*len(inps)}) else torch.save({'inputs':inps,'targets':[0]*len(inps)},gym_env+'.ds')


def array2pt(array):
    return [ torch.tensor(t) for t in array]

def df2ds(df,inputs,target):
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
            'inputs': [ array2pt( [ df[i][r] if i not in in_tables else in_tables[i][df[i][r]] for i in inputs ] ) for r in range(df.shape[0]) ] ,
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
    # image = Image.open(io.BytesIO(image) if isinstance(image,bytes) else image )
    image = Image.open(image)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform_img:
        image = transform(image).unsqueeze(0)
        image = image[:,:3,:,:]
    
    return image

