from feniQS.general.general import CollectPaths

import numpy as np
import yaml

pth_yaml_functions = CollectPaths('./feniQS/general/yaml_functions.py')

def yamlDump_pyObject_toDict(obj, full_name):
    _possible_types = (int, float, dict, list, str, bool, tuple)
    with open(full_name, 'w') as f:
        if isinstance(obj, dict):
            A = obj
        else:
            A = obj.__dict__
        # Due to possible change of A's values (with their __dict__), we get a copy of it:
        AA = {k: v for k, v in A.items()}
        def _check_object_attributes(aa): # Checks types of attributes, and potentially replace them with their dictionary.
            for k in aa.keys():
                if (aa[k] is not None) and (not isinstance(aa[k], _possible_types)):
                    try:
                        aa[k] = aa[k].__dict__
                        _check_object_attributes(aa[k])
                    except:
                        wrn = f"The type of attribute {k} is {type(aa[k])}, which is not supported to be dumped to yaml file."
                        wrn += f"\nAn attempt to instead dump its dictionary (of its own attributes) failed either."
                        raise TypeError(wrn)
        _check_object_attributes(AA)
        yaml.dump(AA, f, sort_keys=False)

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

