import os
import dolfin as df
from feniQS.general.general import CollectPaths, make_path
from feniQS.general.yaml_functions import *

pth_parameters = CollectPaths('./feniQS/general/parameters.py')
pth_parameters.add_script(pth_yaml_functions)

def get_hashable_of_dict(dict_obj):
        """
        keys of the dictionary are put into a tuple and values are so (into another tuple),
        the two tuples are put into a new tuple, too.
            --> The final tuple, which is returned, is then 'hashable'.
        """
        hashables = (int, float, str, bool, type) # tuple is also hashable but as long as its entries are all hashables. So, it is handled recursively similar to list.
        def handle_value(v):
            if not isinstance(v, hashables):
                if isinstance(v, ParsBase):
                    v = v.get_hashable()
                elif isinstance(v, dict):
                    kks, vvs = handle_dict(v.items())
                    v = (tuple(['dict_keys']+[kk for kk in kks]), tuple(['dict_values']+[vv for vv in vvs]))
                elif isinstance(v, list) or isinstance(v, tuple):
                    vt = ()
                    for vv in v:
                        vt += (handle_value(vv), )
                    v = vt
                else:
                    _msg = f"The parameter '{v}' with the type '{type(v).__name__}' is not hashable."
                    _msg += f"\nYou can resolve it by changing the parameter type to either of the hashable types: "
                    _msg += f"{['None'] + [kk.__name__ for kk in hashables]}. The types 'list' and 'dict' are also handled."
                    raise ValueError(_msg)
            return v
        def handle_dict(dict_items):
            keys = []; vals = []
            for k, v in dict_items:
                if v is not None:
                    v = handle_value(v)
                keys.append(k)
                vals.append(v)
            return ((), ()) if len(keys)==0 else zip(*sorted(zip(keys, vals))) # We sort both according to order of keys.
        keys, vals = handle_dict(dict_obj.items())
        # return (tuple(keys), tuple(vals))
        return (tuple(['dict_keys']+[kk for kk in keys]), tuple(['dict_values']+[vv for vv in vals]))

class ParsBase:
    def __init__(self, pars0=None, **pars_dict):
        self.__dict__.update(pars_dict)
        if pars0 is not None:
            self.__dict__.update(pars0.__dict__)
    def get_copy(self):
        _cls = self.__class__
        import copy
        return _cls(**copy.deepcopy(self.__dict__))
            # It is crucial to make copy of dictionary items of the original object (self), otherwise,
            # some attributes (ditionary values) with types such as dictionary would not be copied,
            # instead, the same (dictionary) object would attribute to the self's copy that we create here.
    def yamlDump_toDict(self, _path, _name='parameters'):
        yamlDump_pyObject_toDict(self.get_hardened_dict(), _path + _name + '.yaml')
    def soften(self, softenned_pars_list):
        d = self.__dict__
        for kp in softenned_pars_list:
            d[kp] = df.Constant(d[kp])
    def get_hardened_dict(self):
        d1 = self.__dict__; d2 = {}
        for kp, vp in d1.items():
            if isinstance(vp, df.Constant): # We convert back the softenned parameter (of df.Constant type) to float value.
                d2.update({kp: float(vp.values()[0])})
            else:
                d2.update({kp: vp})
        return d2
    def get_hashable(self):
        return get_hashable_of_dict(dict_obj=self.get_hardened_dict())
    @staticmethod
    def get_merged_hashable(pars_names, list_of_pars):
        """
        Given:
            pars_names = ['name_1', 'name_2', ...]
            list_of_pars = [ParsBase_object_1, ParsBase_object_2, ...]
        , it returns a 2-length tuple (names, hashables) with:
            names = ('merged_parameters_names', 'name_1', 'name_2', ...)
            hashables = ('merged_parameters_hashables', 'hashable_of_ParsBase_object_1', 'hashable_of_ParsBase_object_2', ...)
        .
        """
        assert all(isinstance(p, ParsBase) for p in list_of_pars)
        assert len(list_of_pars)==len(pars_names)
        names = ('merged_parameters_names', ); hashables = ('merged_parameters_hashables', )
        for i, p in enumerate(list_of_pars):
            names += (pars_names[i], ) # append
            hashables += (p.get_hashable(), ) # append
        return (names, hashables)
    @staticmethod
    def get_from_hashable(hashable, return_as='dict'):
        """
        This method does the reverse of either 'ParsBase.get_merged_hashable' or 'ParsBase.get_hashable';
            i.e. it converts back a hashable
                - that is generated from a single ParsBase object or merged from several ParsBase objects -
            into corresponding parameters object either as dictionary (return_as='dict') or python pbject (return_as=a python Class).
            If input 'hashable' comes from a merged case, the final return value is of 'dict' type with keys being respective names of ParsBase objects.
        """
        def handle_value(v):
            if isinstance(v, tuple):
                if len(v)==2 and isinstance(v[0], tuple) and isinstance(v[1], tuple) \
                and v[0][0]=='dict_keys' and v[1][0]=='dict_values':
                    # It has been a dictionary (see ebove in get_hashable)
                    keys = v[0][1:]; vals = v[1][1:]
                    v = get_dict(keys, vals)
                else:
                    # It is a normal tuple, so, every entry (value) is handled individually.
                    v = tuple(handle_value(vv) for vv in v)
            return v
        def get_dict(keys, vals):
            a = dict()
            for i, k in enumerate(keys):
                v = vals[i]
                if k=='merged_parameters_names' and v=='merged_parameters_hashables':
                    for j in range(1, len(keys)):
                        a.update({keys[j]: ParsBase.get_from_hashable(vals[j], return_as)})
                    return_as = 'dict' # for final return (of a) it is dictionary.
                    break
                else:
                    a.update({k: handle_value(v)})
            return a
        assert isinstance(hashable, tuple)
        assert len(hashable)==2
        a = get_dict(keys=hashable[0], vals=hashable[1])
        if isinstance(return_as, str):
            if return_as.lower()=='dict':
                return a
            elif return_as.lower()=='parsbase':
                return ParsBase(**a)
        else:
            try:
                return return_as(**a)
            except:
                raise ValueError(f"Given 'return_as' is not recognized. Use 'dict' or a valid (sub-)class of 'ParsBase'.")
    @staticmethod
    def unique_write(pars_list, pars_names, root, subdir=''):
        """
        This method traces 'unique' realization/storage of a list of instances of ParsBase class.
        'Unique' implies being different w.r.t. at least one attribute of one instance.
        The tracing is according to what have already been stored (as yaml files) in either of the followings:
            - if subdir='':
                stored in 'root',
                with the names: f"{name}_{ID}"
            - if subdir!='':
                stored in 'root' + 'subdir_' + 'ID',
                with the names: f"{name}"
            , where 'name's come from 'pars_names' and 'ID' is identifier (as integer) to any unique set of parameters.
        Returns:
            - the ID corresponding to traced set of parameters,
            - the storage path of set of parameters.
        """
        make_path(root)
        assert all([isinstance(p, ParsBase) for p in pars_list])
        assert len(pars_list)==len(pars_names)
        
        def get_path_and_names(_id):
            if len(subdir)==0:
                _path = f"{root}"
                _names = [f"{n}_{_id}" for n in pars_names]
            else:
                _path = f"{root}{subdir}_{_id}/"
                make_path(_path)
                _names = [f"{n}" for n in pars_names]
            return _path, _names
        
        ## Build identifier_dict
        identifier_dict = dict()
        _ids = []
        if len(subdir)==0:
            for file in os.listdir(root):
                if any([file.endswith(fr) for fr in ['.yaml']]):
                    a2 = file.rfind('.yaml')
                    a1 = file.rfind('_')
                    _id = int(file[a1+1:a2])
                    if _id not in _ids:
                        _ids.append(_id)
        else:
            for ff in os.listdir(root):
                d = os.path.join(root, ff)
                if os.path.isdir(d) and (subdir in ff):
                    a1 = ff.rfind('_')
                    _id = int(ff[a1+1:])
                    if _id not in _ids:
                        _ids.append(_id)
        for _id in _ids:
            pth, files = get_path_and_names(_id)
            if len(files)==1:
                hp = ParsBase(**yamlLoad_asDict(pth + files[0] + '.yaml')).get_hashable()
            else:
                ps = [ParsBase(**yamlLoad_asDict(pth + file + '.yaml')) for file in files]
                hp = ParsBase.get_merged_hashable(pars_names, list_of_pars=ps)
            identifier_dict[hp] = _id
        
        ## Uniquely write new set of parameters
        if len(pars_list) > 1:
            hp = ParsBase.get_merged_hashable(pars_names=pars_names, list_of_pars=pars_list)
        else:
            hp = pars_list[0].get_hashable()
        
        if hp in identifier_dict.keys():
            _id = identifier_dict[hp]
            _msg = f"The parameter set are/is already created and stored in:"
            pp, nns = get_path_and_names(_id)
            for nn in nns:
                _msg += f"\n\t- '{pp}{nn}.yaml'"
            print(f"{_msg} .")
        else:
            _id = 0 if len(identifier_dict)==0 else max(identifier_dict.values()) + 1
            identifier_dict.update({hp: _id})
            pp, nns = get_path_and_names(_id)
            for ni, nn in enumerate(nns):
                pars_list[ni].yamlDump_toDict(_path=pp, _name=nn)
            _msg = f"\nA new (unique) set of parameters are/is stored in:"
            for nn in nns:
                _msg += f"\n\t- '{pp}{nn}.yaml'"
            print(f"{_msg} .")
        pth = root if len(subdir)==0 else f"{root}{subdir}_{_id}/" # storage path of set of parameters
        return _id, pth

class SimulationsTracer():
    def __init__(self, root, simulator, results_writer=None, results_reader=None, outputs_subdir='RESULTs'):
        """
        root : str
            The root directory with respect to which all simulations are done.
        simulator : callable
            Does simulation.
            Input arguments:
                - pth: the root directory that a simulator might wish to get (simulator can potentially write some results as well).
                - all input parameters (key indexed) such as **pars_as_dict
            Return:
                - simulation outputs
        results_writer : callable
            Writes appropriate files containing desired results of a simulation.
            Input arguments:
                - pth: root path for writing result files.
                - simulation_outputs: any kind of python objects as simulation outputs (possibly several objects in a list or dictionary).
                - all input parameters (key indexed) such as **pars_as_dict
            Return:
                - simulation results that have been written (and can be read back later on): any kind of python object (possibly several objects in a list or dictionary).
        results_reader : callable
            Reads results files of a simulation performed before.
            Input arguments:
                - pth: root path where the results files are stored.
                - all input parameters (key indexed) such as **pars_as_dict
            Return:
                - simulation results: any kind of python object (possibly several objects in a list or dictionary).
        outputs_subdir : str, optional
            A sub_directory (under root) where results are stored/written to and read back from. The default is 'RESULTs'.

        __call__
        -------
        The __call__ method of this class has:
            Input arguments:
                - all input parameters (key indexed) such as **pars_as_dict for performing a simulation.
            Return:
                - results: any kind of python object as simulation results, which is either done first or read back from stored results files.
                    (in case of several ones, one could put them in a list or dictionary)
        """
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.root_results = root + outputs_subdir + '/'
        if not os.path.exists(self.root_results):
            os.makedirs(self.root_results)
        
        if results_writer is None:
            print(f"\n--- SimulationsTracer - WARNING ---\n\tNo results_writer is specified (simulation results will not be written).")
            def results_writer(pth, outputs, **pars_as_dict):
                return outputs
        if results_reader is None:
            print(f"\n--- SimulationsTracer - WARNING ---\n\tNo results_reader is specified. Results of the simulations already done before will not be read back.")
            def results_reader(pth, **pars_as_dict):
                pass
        
        self.simulator = simulator
        self.results_writer = results_writer
        self.results_reader = results_reader
        self.identifier_file = self.root + 'results_identifier' + '.yaml'
        self.identifier_dict = dict()
        
    def __call__(self, **pars_as_dict):
        """
        pars_as_dict:
            A dictionary like:
                {'model_pars': ParsBase_obj_1
        """
        if os.path.exists(self.identifier_file):
            self.identifier_dict = yamlLoad_asDict(self.identifier_file)
        assert len(pars_as_dict) > 0
        assert all([isinstance(p, ParsBase) for p in pars_as_dict.values()])
        if len(pars_as_dict) > 1:
            hp = ParsBase.get_merged_hashable(pars_names=list(pars_as_dict.keys()), list_of_pars=list(pars_as_dict.values()))
        else:
            hp = list(pars_as_dict.values())[0].get_hashable()
        if hp in self.identifier_dict.keys():
            res_id = self.identifier_dict[hp]
            results = self._read_results(res_id, **pars_as_dict)
        else:
            if len(self.identifier_dict)==0:
                new_id = 0
            else:
                new_id = max(self.identifier_dict.values()) + 1
            pth = self.root_results + f"results_{new_id}/"
            assert (not os.path.exists(pth)) # must be a new path/simulation results
            os.makedirs(pth)
            outputs = self.simulator(pth, **pars_as_dict)
            results = self.results_writer(pth, outputs, **pars_as_dict)
            print(f"\n--- SimulationsTracer ---\n\tNew simulation done and results stored in {pth}.")
            self.identifier_dict.update({hp: new_id})
            yamlDump_pyObject_toDict(self.identifier_dict, self.identifier_file) # overwrite simulation results
            print(f"\n--- SimulationsTracer ---\n\tSimulation results identifier file '{self.identifier_file}' updated.")
        return results
    
    def get_from_identifier(self, ID):
        self.identifier_dict = yamlLoad_asDict(self.identifier_file)
        hp = None
        for k, v in self.identifier_dict.items():
            if v == ID:
                hp = k
        if hp is None:
            print(f"\n--- SimulationsTracer ---\n\tNo simulation found for the given ID={ID}.")
            return None
        else:
            pars_as_dict = ParsBase.get_from_hashable(hp)
            return self._read_results(res_id=ID, **pars_as_dict)
    
    def _read_results(self, res_id, **pars_as_dict):
        pth = self.root_results + f"results_{res_id}/"
        print(f"\n--- SimulationsTracer ---\n\tThe simulation is already done before. ... reading the results back from:\n\t\t'{pth}'.\n")
        results = self.results_reader(pth, **pars_as_dict)
        print(f"\n--- SimulationsTracer ---\n\tThe results read back from:\n\t\t'{pth}'.\n")
        return results