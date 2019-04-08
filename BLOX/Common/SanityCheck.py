
REQS = {
    "Epochs":1,
    "SaveEvery":-1,
    "BatchSize":32,
    "DataSet":"test.ds",
    "FileExt":"test.pt",
    "Verbose":False,
    "TensorboardX":{
        
    }
}

def CheckConfig(self,config):
    for k in REQS.keys():
        if k not in config:config[k] = REQS[k]
    return config