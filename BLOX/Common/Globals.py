
import importlib,inspect
def load_dynamic_modules(module):
    try:
        imported = import_module(module)
    except:
        imported = __import__(module)
    table = {}
    for c in inspect.getmembers(imported, inspect.ismodule):
        try:
            table[c[0]] = getattr(c[1],c[0])
        except Exception as e:pass
    return table
USER_DEFINED = {}
PREDEFINED = load_dynamic_modules('BLOX.Modules')

# PATH

# def get_global(key):
