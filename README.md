# B L O X
### Neural Network building blocks. A simple and extensible wrapper around pytorch

BLOX is a JSON based configuration library meant to easy and automate the development of neural network models using the PyTorch library.
In can be applied in many different ways. For example, if you want to only define and compile a neural network:

```python
import json
import torch
from BLOX import Compile

'''
    The Compiler can consume either a dictionary 
    or the path to a JSON net file, which is demonstrated below.
'''
json_string = """{
    "Layers":[
            {
                "DEF":"MyFirstBlock"
            },
            {
                "Other":[
                    "Conv1d",
                    {
                        "in_channels":128,
                        "out_channels":32,
                        "kernel_size":4
                    }
                ]
            }
    ],
    "Notes":"",
    "Name":"",
    "DEFS":{
            "BLOCKS":{
                "MyFirstBlock": [
                    {
                        "Linear":{
                            "in_features":64,
                            "out_features":32
                        }
                    },
                    {
                        "Act":"ReLU"
                    }
                ]
            },
            "FILES":[
            ]
        }
}"""

loaded = json.loads(json_string)


model = Compile(loaded)
x = torch.randn(1,128)
y = model(x)

```


