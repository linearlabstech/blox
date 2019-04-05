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
import torch,numpy


class Jsonify:

    def __init__(self):pass

    def __call__(self,x):
        def handle_data(data):
            if isinstance(data,torch.Tensor):return data.cpu().data.numpy().tolist()
            elif isinstance(data,numpy.ndarray):data.tolist()
            elif isinstance(data,dict):return dict(
                                                zip(
                                                    data.keys(),
                                                    [handle_data(v) for v in data.values() ]
                                                )
                                            )
            return data
        return handle_data(x)