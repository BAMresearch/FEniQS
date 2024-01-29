from feniQS.general.pydoit import DoitTaskManager
from feniQS.virt_lab.model_INPUTs import *

class TaskGenerateVirtualDataSetsPerfect(DoitTaskManager):
    def __init__(self, root_input_models, root_virtual_data):
        DoitTaskManager.__init__(self)
        self.root_input_models = root_input_models # input yaml files (of parameters) to model simulation
        self.root_virtual_data = root_virtual_data # root of where virtual data is stored.

        ## IMPORTANT: This task potentially contains several sub-tasks in parallel.
        self.has_subTasks = True
        ### INPUTs
        self.data_generator_callables = {}
        ## Input to such callables is just a number as data_ID, which refers to
        # a certain set of INPUT parameters stored before.
        self.selected_data_IDs = {}
        ## The data is generated as regards to the selected data_ID(s), each refering
        # to a certain set of INPUT parameters stored before.
        self.data_generator_targets_callables = {}
        ## Per model name, is a callable that returns all target files.
        # Input to such callables is just the data_ID.
        self.pths_dep_files = {}
        ## Per model name, is a list of dependency files regarding the scripts (NOT for files of input parameters).
        self.tasks_dep = []

    def add_data_sets(self, model_name, inputs_IDs \
                      , data_generator_callable, data_generator_targets_callable \
                      , pths_dep_files):
        if isinstance(inputs_IDs, int):
            inputs_IDs = [inputs_IDs]
        self.data_generator_callables[model_name] = data_generator_callable
        self.data_generator_targets_callables[model_name] = data_generator_targets_callable

        if model_name not in self.selected_data_IDs.keys():
            self.selected_data_IDs[model_name] = []  # initiation
        for _id in inputs_IDs:
            if _id not in self.selected_data_IDs[model_name]:
                self.selected_data_IDs[model_name].append(_id)
        self.pths_dep_files[model_name] = pths_dep_files
        
    
    def get_subTasks_names(self):
        for mn, data_IDs in self.selected_data_IDs.items():
            for data_ID in data_IDs:
                yield f"Model_name:'{mn}', Input_pars_ID:{data_ID}"

    def get_file_dep(self):
        for mn, data_IDs in self.selected_data_IDs.items():
            for data_ID in data_IDs:
                dep_files = get_input_file_model(model_name=mn, inputs_ID=data_ID, which_input="all" \
                                                 , root_input_models=self.root_input_models)
                yield self.pths_dep_files[mn] + dep_files

    def get_actions(self):
        for mn, data_IDs in self.selected_data_IDs.items():
            f = self.data_generator_callables[mn]
            for data_ID in data_IDs:
                yield [(f, [data_ID])]

    def get_targets(self, only_files=False):
        for mn, data_IDs in self.selected_data_IDs.items():
            for data_ID in data_IDs:
                _name = f"{mn}_{data_ID}"
                _path = f"{self.root_virtual_data}{_name}/"
                target_files = self.data_generator_targets_callables[mn](
                    data_ID=data_ID
                )
                if only_files:
                    yield target_files
                else:
                    yield [_path] + target_files
