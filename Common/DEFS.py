
import os,yajl as json
import inspect
from importlib import import_module

# BLOCKS = load('DEFS.json')

def load(fname):
    modules = [fname["Module"]]#open(fname,'r').read()
    table = {}
    args = fname["Args"]
    for i,module in enumerate(modules):
        imported = import_module(module)
        # print(inspect.getmembers(imported))
        for c in inspect.getmembers(imported, inspect.isclass):
            # print(c[0],getattr(imported,c[0]))
            try:
                table[c[0]] = c[1](**args)
            except:
                pass
    return table

