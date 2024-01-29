import os
from feniQS.general.parameters import ParsBase
from feniQS.general.yaml_functions import *
from feniQS.problem.model_time_varying import (
    QuasiStaticSolveOptions,
    pth_model_time_varying,
)
from feniQS.general.general import CollectPaths

pth_model_INPUTs = CollectPaths("./feniQS/virt_lab/model_INPUTs.py")
pth_model_INPUTs.add_script(pth_yaml_functions)
pth_model_INPUTs.add_script(pth_model_time_varying)


def get_inputs_of_model(model_name, inputs_ID, which_input, root_input_models, pars_cls=None):
    """
    - inputs_ID: an ID specifying a certain set of all model inputs, including:
        'pars'
        'solve_options'
    - which_input: must be either of those 2 cases.
    - pars_cls: must be provided only if which_input=='pars'.
    """
    if ("pars" in which_input.lower()) and pars_cls is None:
        raise ValueError(f"Class of input parameters (of model) must be specified.")
    file = get_input_file_model(model_name=model_name, inputs_ID=inputs_ID \
                                , which_input=which_input, root_input_models=root_input_models)
    _cls = {"pars": pars_cls, "solve_options": QuasiStaticSolveOptions}[
        which_input.lower()
    ]
    return _cls(**yamlLoad_asDict(file))


def get_input_file_model(model_name, inputs_ID, which_input, root_input_models):
    # The following file path is in view of:
    # the implementation of '.feniQS.general.parameters.ParsBase.unique_write' method,
    # how the parameters are written.
    _path = f"{root_input_models}{model_name}/{model_name}_{inputs_ID}/"
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"No input parameter sets have been generated in '{_path}'."
        )
    if which_input.lower() == "all":
        return [f"{_path}{aa}" for aa in os.listdir(_path)]
    else:
        return f"{_path}{which_input}.yaml"


def write_inputs_model(inputs_model_getter, root_input_models, _return="ID"):
    """
    inputs_model_getter:
        A callable with no input, which returns a dictionary whose keys must include:
            - 'model_name'
            - 'pars_struct'
            - 'pars'
            - 'solve_options'
    """
    required_inputs = ['model_name', 'pars_struct', 'pars', 'solve_options']
    model_inputs_dict = inputs_model_getter()
    if not isinstance(model_inputs_dict, dict) or any([k not in model_inputs_dict.keys() for k in required_inputs]):
        raise ValueError(f"The callable for getting model inputs must return a dictionary whose keys include {required_inputs}.")

    model_name = model_inputs_dict['model_name']
    pars_struct = model_inputs_dict['pars_struct']
    pars = model_inputs_dict['pars']
    solve_options = model_inputs_dict['solve_options']
    if not isinstance(pars, list):
        pars = [pars]
        pars_struct = [pars_struct]
    if not isinstance(solve_options, list):
        solve_options = [solve_options]
    inputs_IDs = []
    paths = []
    for p in pars:
        for so in solve_options:
            inputs_ID, path = ParsBase.unique_write(
                pars_list=[p, so],
                pars_names=["pars", "solve_options"],
                root=f"{root_input_models}{model_name}/",
                subdir=model_name,
            )
            inputs_IDs.append(inputs_ID)
            paths.append(path)
            for k, v in model_inputs_dict.items():
                if k not in required_inputs:
                    if isinstance(v, np.ndarray):
                        yamlDump_array(a=v, full_name=f"{path}{k}.yaml")
    if _return == "ID":
        return inputs_IDs, paths
    else:
        return pars_struct, pars, solve_options