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
