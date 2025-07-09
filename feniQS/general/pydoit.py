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


def run_pydoit_task(tasks, basename
                    , verbosity=2, reset_dep=False
                    , dep_file=None):
    """
    Source:
        https://pydoit.org/extending.html#example-pre-defined-task
    """
    import types
    if not isinstance(tasks, list):
        tasks = [tasks]
    def _adjust(task):
        if isinstance(task, DoitTaskManager):
            t = task.get_task_dict()
        elif callable(task):
            t = task()
        else: # is a dictionary
            t = {k:v for k,v in task.items()} # Much safer to get copy/reference (due to possible change of task names)
        if not (isinstance(t, list) or isinstance(t, types.GeneratorType)):
            t = [t]
        return t
    from doit.task import dict_to_task
    from doit.cmd_base import TaskLoader2
    from doit.doit_cmd import DoitMain
    class MyLoader(TaskLoader2):
        def setup(self, opt_values):
            pass
        def load_doit_config(self):
            conf = {'verbosity': verbosity}
            if dep_file is not None:
                conf['dep_file'] = dep_file
            return conf
        def load_tasks(self, cmd, pos_args):
            task_list = []
            for task in tasks: # by itself can have subtasks or a single task
                t = _adjust(task) # get genarator of subtasks, or a list of a single task
                for td in t:
                    if basename!='':
                        td['name'] = f"{basename}:{td['name']}"
                    task_list.append(dict_to_task(td))
            return task_list
    doit_instance = DoitMain(MyLoader())
    if reset_dep:
        doit_instance.run(['reset-dep'])
        doit_instance.run(['run'])
    else:
        doit_instance.run(['run'])