import sys
if './' not in sys.path:
    sys.path.append('./')

from feniQS.general.general import CollectPaths

import numpy as np
import yaml

pth_yaml_functions = CollectPaths('yaml_functions.py')

def yamlDump_pyObject_toDict(obj, full_name):
    _possible_types = (int, float, dict, list, str, bool, tuple)
    with open(full_name, 'w') as f:
        if isinstance(obj, dict):
            aa = obj
        else:
            aa = obj.__dict__
        ## CHECK types of object's attributes
        for k in aa.keys():
            if aa[k] is not None:
                if not isinstance(aa[k], _possible_types):
                    raise TypeError(f"The type of attribute {k} is {type(aa[k])}, which is not supported to be dumped to yaml file.")
        yaml.dump(aa, f, sort_keys=False)

def yamlLoad_asDict(full_name):
    with open(full_name) as f:
        loaded = yaml.load(f, Loader=yaml.FullLoader)
    return loaded

def yamlDump_array(a, full_name):
    with open(full_name, 'w') as f:
        yaml.dump(a.tolist(), f)

def yamlLoad_array(full_name):
    with open(full_name) as f:
        loaded = yaml.load(f, Loader=yaml.FullLoader)
    return np.array(loaded)

