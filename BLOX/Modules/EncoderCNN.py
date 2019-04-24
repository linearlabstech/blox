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
import torchvision.models as models
from torch import nn
import torch
class EncoderCNN(nn.Module):
    def __init__(self, embed_size,model_type="resenet-50"):
        """
            Load the pretrained net and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        resnet = getattr(models,model_type)(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """
            Extract feature vectors from input images.
        """
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        try:
            features = self.bn(self.linear(features))
        except:
            features = self.linear(features)
        return features