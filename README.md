<p align="center">
  <img src="img/BLOX.png"/>
</p>


### Neural Network building blocks. A simple and extensible wrapper around pytorch

BLOX is a JSON structre based configuration library meant to accelerate and automate the development of neural network models using the PyTorch library.
Because of the JSON configuration it allows for ease of integration into web applications.


## Table Of Contents
1. [Basic Usage](#using)
2. [Creating Networks in BLOX](#creating)
3. [Defining BLOX](#defining)
4. [Importing Modules](#importing)
5. [Extending BLOX Modules](#extending)
6. [DataSet Creation](#datasets)
7. [PlaceHolders](#placeholders)
8. [Training With BLOX](#training)
9. [Tensorboard](#tensorboard)

### TODOs

1. We will soon be adding Kubernetes and RabbitMQ support. This way you will be able to quickly train, test and deploy. 
2. Adding automatic data cleaning support.
3. Add more TensorboardX support.

## <a name="using"></a> Basic Usage
The BLOX framework can be applied in many different ways. For example, if you want to only define and compile a neural network, you can dreate a config and pass it to the compiler, like so.

```python
import json
import torch
from BLOX import Compile

'''
    The Compiler can consume either a dictionary 
    or the path to a BLOX file, which is demonstrated below.
'''
json_string = """{
    "BLOX":[
            {
                "DEF":"MyFirstBlock"
            },
            {
                "Other":[
                    {
                        "Conv1d": {
                            "in_channels":128,
                            "out_channels":32,
                            "kernel_size":4
                        }
                    }
                ]
            }
    ],
    "Notes":"",
    "Name":"",
    "DEFS":{
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
     "IMPORTS":[
    ]
}"""

loaded = json.loads(json_string)

model = Compile(loaded)
x = torch.randn(1,128)
y = model(x)

```
The blox you define are the classes that currently exist in the pytorch library. The JSON dictionary value that is adjacent to the definintion requires accuracte keyword arguments.
For example, in the above, the `Conv1d` that is used is generated with dictionary values it holds.


## <a name="creating"></a> Creating Networks in BLOX

As you could probably tell from the examples above, BLOX allows you to define network architectures via a JSON format in the `DEFS` section.
Although the simpliest thing would be to define all of your networks in one file, that can quickly become cumbersome to navigate and use.
Because of this, the BLOX library allows for BLOX definitions is seperate files, which can be imported.
So if you have a BLOX DEF, like so
```JSON
{
    "MyDefBlock": [
        {
            "Linear":{
                "in_features":64,
                "out_features":32
            }
        },
        {
            "Act":"ReLU"
        }
    ],
    "MyOtherDefBlock": [
        {
            "Linear":{
                "in_features":64,
                "out_features":32
            }
        },
        {
            "Act":"Sigmoid"
        }
    ],
}
```
which you can save as `my_defs.blx` and load via the `IMPORTS` section by simply passing the path to the `.blx` file.
Finally, this can be included in your main block like so.
```JSON
{
    "BLOX":[
            {
                "DEF":"MyDefBlock"
            },
            {
                "DEF":"MyDefBlock"
            }
            {
                "DEF":"MyOtherDefBlock"
            }
    ],
    "Notes":"",
    "Name":"",
    "DEFS":{
        },
    "IMPORTS":[
        "my_defs.blx"
    ]
}

```

As you are probably able to infere, when you define an architecture there are three keywords you need to pay attention to.

1. The `Linear` keyword defines a weight matrix size
2. The `Act` keyword defines the nonlinear function. We support any Pytorch nonlinearity
3. The `Other` keyword is used to define basically everything else, for example convolution layers.

#### Spelling and case count!

see [IMPORTS](#importing) for more details.

For those of you who are lazy, and want to repeat multiple blox, we've built in the `REPEAT` keyword for that.
So, for example we want ot implement a deep architecure, you could simply repeat the block.
```JSON
{
    "BLOX":[
            {
                "REPEAT":{
                    "BLOCK":"MyDefBlock",
                    "REPS":8
                }
            },
            {
                "DEF":"MyOtherDefBlock"
            }
    ],
    "Notes":"",
    "Name":"",
    "DEFS":{
        },
    "IMPORTS":[
        "my_defs.blx"
    ]
}

```
Using the `REPS` keyword will sequentially repeat the block the number of times you tell it to.

<!-- Cool, that's awesome, now we can make DNNs, but what about residual nets? Could we make those? -->

## <a name="importing"></a> Importing Modules
BLOX support multiple import types. Specifically, you may import JSON & Python files. 

#### JSON
In order to import JSON/blx files, you must specifiy the extention. For example, in you `IMPORTS` subsection you only need to specify the relative path to the file.

```JSON
{
    ...,

    "IMPORTS":[
        "my_defs.blx",
        "more_defs.json"
    ]
}

```

When importing BLOX defs, you only need to provide the `DEFS` section. For example, saving the following 

```JSON
{
    "MyDefBlock": [
        {
            "Linear":{
                "in_features":64,
                "out_features":32
            }
        },
        {
            "Act":"ReLU"
        }
    ],
    "MyOtherDefBlock": [
        {
            "Linear":{
                "in_features":64,
                "out_features":32
            }
        },
        {
            "Act":"Sigmoid"
        }
    ],
}

```
as `my_defs.blx` and importing into another `blx` file will allow access to both `MyDefBlock` and `MyOtherDefBlock`.


#### Pyhton Modules
If you wish to extend the IMPORTS with custom python modules, you must specify the relative via python syntax ( using "." for directories and excluding the ".py" extension). For example,

```JSON
{
    ...,

    "IMPORTS":[
        "my_defs.blx",
        "more_defs.json",
        "my.python.module"
    ]
}
```
All of the definitions whether from python or blox, should be assumed accessible in the 

## <a name="extending"></a> How to extend BLOX Modules
Let's say you've created a new neural network like below, 
```python
import torch
from torch import nn

class Net(nn.Module):
    
    def __init__(self,n_in,n_h,n_out):
        super(Net,self).__init__()
        self.model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
    
    def forward(self,x):
        return self.model(x)

```
and in a source file name `Net.py`.
Simply run the `add2blx Net.py` and now you can access your network in two ways.

1. Through the `BLOX` Module via source code, like so
    ```python
    from BLOX.Modules import Net
    model = Net(32,4,32)
    x = torch.randn(1,32)
    y = model(x)
    ```
2. Through a BLOX definition, like so
    ```JSON
    {
        "BLOX":[
            {
                "DEF":"Net"
            },
        ],
    }
    ```

## <a name="datasets"></a> Creating DataSets

BLOX DataSet objects are built to support the development of easy training regiments.
These objects can be instatiated by either passing a string (the path to location of the dataset) or a dictionary object.

```python
from BLOX import DataSet

data = DataSet('path/to/file.ds')

# or

data = DataSet({
    'inputs':[...],
    'targets':[...]
})

# both the 'inputs' & 'targets' keywords are required

```

If you wish to work with tabular data and use a CSV file, you can integrate the Pandas library as so.

```python
import pandas as pd
from BLOX import df2ds 

df = pd.read_csv('path/to/your.csv')

ds = df2ds(df,['input','columns'],'target column',)


```

The DataSet object automatically divides into training and development splits (85/15% respectively). You can iterate over these cuts by calling thier respecive methods. For the training split call `.train()` and for the development split call `.dev()`. These are also automatically set to a GPU if the device is available. 

## <a name="placeholders"></a> PlaceHolders

The PlaceHolder Object is a variable holder that can be used for residual connections. The object support the four arethmatic operations 

1. Addition (+)
2. Subtraction (-)
3. Multiplication (*)
4. Division (/)

To impliment the PlaceHolder in your BLOX definition, just simply call it as it is already defined in the Modules

```JSON
{
    "BLOX":[
        {
            "DEF":"PlaceHolder" <- store the variable, no compute done on it
        },
        {
            "DEF":"MoreOperationsBLOX"
        },
        {
            "DEF":"PlaceHolder" <- get the original variable passed in before and compute on the input
        }
    ]
}
```

## <a name="training"></a> Training with BLOX

Training networks using BLOX can be done through your typical training pipeline, only using BLOX to assemble the network, or you can impliment the `Trainer` module.
To implement the `Trainer` class, import and pass the config.

```python
import json
from BLOX import Trainer

path = 'path/to/your/config'
Trainer(path).run()

# or

json_string = """{
    "Nets":{
        "Frozen":"freeze_net.json"
        "ConfigNet":"net.json"
    },
    "Optimizer":{
        "Params":[
            "ConfigNet"
        ],
        "Kwargs":{
            "lr":0.001
        },
        "Algo":"SGD"
    },
    "Loss":{
        "Algo":"L1Loss",
        "Kwargs":{}
    },
    "Epochs":50,
    "SaveEvery":50,
    "BatchSize":32,
    "DataSet":"test.ds",
    "FileExt":".pt",
    "Verbose":true,
    "TensorboardX":{
        "LogEvery":10,
        "Dir":"runs/",
        "SaveGraphs":true,
        "Log":{
            "Loss"
        }
    }
}"""

loaded = json.loads(json_string)

Trainer(loaded).run()

```

The confinguration for the `Params` allows you to freeze diffrent blox during training. 
For example, in the config shown above, During training only the "ConfigNet" will be optimized.
This is ensured because only "ConfigNet" is provided in the `Params` section.


## <a name="tensorboard"></a> TensorboardX

In case you noticed above, yes we provide tensorboard support through TensorboardX.

we support the logging of multiple data types. For the "LogEvery" schedule we support:

* Loss

<a href="https://www.freepik.com/free-photos-vectors/background">Background vector created by freepik - www.freepik.com</a>