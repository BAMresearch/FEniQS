from feniQS.general.parameters import *
from feniQS.general.general import *
from feniQS.problem.problem import *
from feniQS.problem.time_stepper import *
from feniQS.general.yaml_functions import *

pth_model_time_varying = CollectPaths('model_time_varying.py')
pth_model_time_varying.add_script(pth_general)
pth_model_time_varying.add_script(pth_problem)
pth_model_time_varying.add_script(pth_time_stepper)
pth_model_time_varying.add_script(pth_yaml_functions)

class QuasiStaticSolveOptions(ParsBase):
    def __init__(self, solver_options=None, **options_dict):
        ## SOLVER
        if solver_options is None:
            solver_options = get_fenicsSolverOptions()
        self.solver_options = solver_options # A dictionary; i.e. get_fenicsSolverOptions()
        if len(options_dict)==0: # Default values are set
            ## TIME (general)
            self.t_start = 0.
            self.t_end = 1.
            self.checkpoints = [] # determining LOAD-STEPs
            self.dt = None
            self.dt_max = 1.0
            self.dt_min = 1e-6
            ## TIME (adaptive)
            self.increase_num_iter = 8
            self.increase_factor = 1.2
            ## POST-PROCESS
            self.reaction_places = []
            ## OTHERs
            self._reset = True
            self._plot = True
        else: # Get froma a dictionary
            self.__dict__.update(options_dict)
        
    def yamlDump_toDict(self, _path, _name='solve_options'):
        ParsBase.yamlDump_toDict(self, _path=_path, _name=_name)

class QuasiStaticModel:
    """
    This is a base class, whose methods "establish_model" and "build_solver" must be overwritten.
    Also, some other methods can be over- or further- written; e.g.: set_pps, solve.
    """
    def __init__(self, pars, softenned_pars=[], _path=None, _name='QuasiStaticModel'):
        self.pars = pars
        self.soften_pars(softenned_pars)
        self._name = _name
        self._path = _path
        
        self.fen = None
        self.established = False
        self.time_varying_loadings = {}
        self.establish_model()
        
        self.solver_built = False
        
        ## PRE-PROCESS
        self.prep_methods = [] # methods called prior to every time-step (e.g. for adjusting loads/BCs) --> See TimeStepper
        self.prep_args = {} # optional arguments (dictionary) to the prep_methods
        ## POST-PROCESS
        self.pps = []
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value) # standard self.name=value
        if name=='_path': # If we change the path, we then update the logger, accordingly.
            if self._path is None:
                self._path = './QS_MODELs/' + self._name + '/'
            self._make_logger_and_path()
        elif name=='penalty_dofs' and hasattr(self, 'fen'):
            self.fen.penalty_dofs = value
    
    def soften_pars(self, softenned_pars_list):
        self.softenned_pars = softenned_pars_list
        self.pars.soften(self.softenned_pars)
        
    def _make_logger_and_path(self):
        make_path(self._path)
        self.logger = LoggerSetup(self._path + self._name + '.log')
    
    def revise_BCs(self, remove=False, new_BCs=[], _as='hom'):
        try:
            self.fen.revise_BCs(remove=remove, new_BCs=new_BCs, _as=_as)
            _msg = "WARNING: The BCs of the model has been modified. Do consider rebuilding the solver given your desired solve_options."
            self.logger.console.warning(_msg)
            self.logger.file.warning(_msg)
        except Exception as e:
            raise AttributeError(f"The model has no fenics object yet. Reconsider establish_model routine.\n{e}")
    
    def establish_model(self): # must be overwritten
        ### MESH ###
        ### MATERIAL ###
        ### FEniCS PROBLEM ###
        ### BCs and LOADs ###
        pass
        
    def build_solver(self, solve_options): # must be overwritten
        ### BUILD SOLVER ###
        pass
    
    def set_pps(self, solve_options):
        ### POST-PROCESS ###
        pass

    def solve(self, solve_options, other_pps=[], _reset_pps=True, write_pars=True):
        if not self.established:
            raise NotImplementedError("No FEniCS model has yet been built. "
                + "Consider overwriting method 'establish_model' in the model class.")
        else:
            if isinstance(solve_options, dict):
                solve_options = QuasiStaticSolveOptions(**solve_options)
            if not self.solver_built:
                self.build_solver(solve_options)
            if not self.solver_built:
                raise NotImplementedError("No solver has yet been built. "
                    + "Consider overwriting method 'build_solver' in the model class.")
            t_end = solve_options.t_end
            checkpoints = solve_options.checkpoints
            t_start = solve_options.t_start
            dt = solve_options.dt
            increase_num_iter = solve_options.increase_num_iter
            increase_factor = solve_options.increase_factor
            dt_min = solve_options.dt_min
            dt_max = solve_options.dt_max
            if solve_options._reset:
                self.fen.reset_fields()
            # Post-processing
            if _reset_pps:
                self.set_pps(solve_options)
            self.pps += other_pps
            
            # define time-stepper
            ts = TimeStepper(self.fen.solve, self.pps, \
                             time_adjusting_methods=self.prep_methods, time_adjusting_args=self.prep_args, \
                             logger=self.logger.file, solution_field=self.fen.get_F_and_u()[1], \
                             increase_num_iter=increase_num_iter, increase_factor=increase_factor, _dt_min=dt_min, _dt_max=dt_max)
            # solve
            if dt==None:
                dt = t_end / 50.
            iterations = ts.adaptive(t_end=t_end, t_start=t_start, dt=dt, checkpoints=checkpoints)
            if solve_options._plot:
                fig = plt.figure()
                plt.plot(ts.ts, iterations, marker='.')
                plt.title('Number of iterations over time\nSum = ' + str(sum(iterations)) + ', # time steps = ' + str(len(ts.ts)))
                plt.xlabel('t')
                plt.ylabel('Iterations number')
                plt.savefig(self._path + 'iterations.png', bbox_inches='tight', dpi=300)
                plt.show()
            if write_pars:
                self.yamlDump_pars()
                yamlDump_pyObject_toDict(solve_options, self._path + 'solve_options.yaml')
            
            return ts, iterations, solve_options
    
    def yamlDump_pars(self, file_name=None):
        d2 = self.pars.get_hardened_dict()
        if file_name is None:
            file_name = self._path + 'pars.yaml'
        yamlDump_pyObject_toDict(d2, file_name)
    
    def get_reaction_dofs(self, reaction_places):
        pass