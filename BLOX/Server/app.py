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

import json
from flask import Flask, request, send_file, jsonify
from flask import render_template
# import tasks

from gunicorn.errors import ConfigError
from gunicorn.app.base import BaseApplication
from gunicorn import util
from flask.views import MethodView

class StandaloneApplication(BaseApplication):

    def __init__(self, app, options={'bind': '%s:%s' % ('127.0.0.1', '5656')},client=None):
        self.options = options or {}
        self.application = app
        super(StandaloneApplication, self).__init__()
        self._client=client
        self.error_handlers = {}


    def register_error_handler(self,name,func):
        self.error_handlers[name] = func

    def set_client_model(self,client):
        self._client = client

    @property
    def client(self):
        if self._client is None:raise AttributeError('You need to set the standalone apps\'s client')
        return self._client

    def register_endpoint(self,url,name,client,arg,preprocess=None,init={}):
        class Func(MethodView):

            def post(self):
                data = request.form.to_dict() 
                data = data if len(data) > 0 else request.get_json()
                if not self.client.initialized:self.client = self.client()
                resp = self.client( preprocess(data[arg] ) if preprocess else data[arg]).decode()
                try:self.client.connection.close()
                except Exception as e:print(e)
                return resp              

    def load_config(self):
        config = dict([(key, value) for key, value in self.options.items()
                       if key in self.cfg.settings and value is not None])
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

def register_endpoint(app,url,name,client,arg,preprocess=None):
    def func():
        data = request.form.to_dict() 
        data = data if len(data) > 0 else request.get_json()
        resp = client( preprocess(data[arg] ) if preprocess else data[arg]).decode()
        return resp
    app.add_url_rule(
            url,
            name,
            func,
            methods=['POST']
    )

def create_app():
    return Flask('BLOX')