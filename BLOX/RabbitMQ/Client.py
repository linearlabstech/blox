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
import pika
import uuid
import time
import torch,json
class Client(object):
    response = None
    corr_id = None
    def __init__(self,host='localhost',queue='default'):

        self.queue = queue

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))

        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue,exclusive=True)

        self.callback_queue = result.method.queue

        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def __call__(self, n,ex=''):

        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange=ex,
                                   routing_key=self.queue,
                                   properties=pika.BasicProperties(
                                         reply_to = self.callback_queue,
                                         correlation_id = self.corr_id,
                                         ),
                                   body=str(n) )
        while self.response is None:
            self.connection.process_data_events()
        return self.response