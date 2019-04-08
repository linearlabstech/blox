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
import torch
import json
from ..Modules.Jsonify import Jsonify 
from ..DataSet.DataTools import img2tensor
import binascii
import io,ast
jsonifier = Jsonify()
def worker(HOST,QUEUE,PIPELINE):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=HOST))

    channel = connection.channel()

    channel.queue_declare(queue=QUEUE)

    def on_request(ch, method, props, x):
        return_error = False
        try:
            x = ast.literal_eval(x.decode())
            x = torch.tensor( [[x] ] )
            for f in PIPELINE:x = f(x)
        except Exception as e:
            print(e)
            return_error = True
        ch.basic_publish(exchange='',
                        routing_key=props.reply_to,
                        properties=pika.BasicProperties(correlation_id = \
                                                            props.correlation_id),
                        body=json.dumps( jsonifier(x) ) if not return_error else 'THERE WAS AN ERROR PROCESSING YOUR REQUEST' )
        ch.basic_ack(delivery_tag = method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_request, queue=QUEUE)
    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
    # except Exception as e:print(e)
    # connection.close()