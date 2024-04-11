class DoitTaskManager:
    """
    A base class for handling pydo stuff.
    """
    def __init__(self):
        self.has_subTasks = False
        self.tasks_dep = [] # globally for all sub_tasks
    @property
    def num_sub_tasks(self):
        if self.has_subTasks:
            return len([t for t in self.get_targets()])
        else:
            return 0
    def get_task_dict(self):
        fds = self.get_file_dep()
        acs = self.get_actions()
        ts = self.get_targets()
        uds = self.get_uptodate()
        if self.has_subTasks:
            ns = self.get_subTasks_names()
            for n, fd, a, t, ud in zip(ns, fds, acs, ts, uds):
                yield {'name': n,
                       'file_dep': fd,
                       'actions': a,
                       'targets': t,
                       'task_dep': self.tasks_dep,
                       'uptodate': ud,
                       }
        else:
            return {'file_dep': fds,
                    'actions': acs,
                    'targets': ts,
                    'task_dep': self.tasks_dep,
                    'uptodate': uds,
                    }
    def get_subTasks_names(self): # only relevant if has_subTasks==True
        raise NotImplementedError(f"Overwrite the implementation of 'get_subTasks_names'.")
    def get_file_dep(self):
        raise NotImplementedError(f"Overwrite the implementation of 'get_file_dep'.")
    def get_actions(self):
        raise NotImplementedError(f"Overwrite the implementation of 'get_actions'.")
    def get_targets(self, only_files=False):
        raise NotImplementedError(f"Overwrite the implementation of 'get_targets'.")
    def get_uptodate(self):
        nst = self.num_sub_tasks
        return [[True] for i in range(nst)]


def run_pydoit_task(dict_tasks, basename='', verbosity=2):
    """
    Source:
        https://pydoit.org/extending.html#example-pre-defined-task
    """
    import types
    if isinstance(dict_tasks, DoitTaskManager):
        dict_tasks = dict_tasks.get_task_dict()
    elif callable(dict_tasks):
        dict_tasks = dict_tasks()
    if not (isinstance(dict_tasks, list) or isinstance(dict_tasks, types.GeneratorType)):
        dict_tasks = [dict_tasks]
    from doit.task import dict_to_task
    from doit.cmd_base import TaskLoader2
    from doit.doit_cmd import DoitMain
    class MyLoader(TaskLoader2):
        def setup(self, opt_values):
            pass
        def load_doit_config(self):
            return {'verbosity': verbosity,}
        def load_tasks(self, cmd, pos_args):
            task_list = []
            for td in dict_tasks:
                if basename!='':
                    td['name'] = f"{basename}:{td['name']}"
                task_list.append(dict_to_task(td))
            return task_list
    DoitMain(MyLoader()).run([])