from feniQS.structure.helper_mesh_fenics import *
from feniQS.structure.helper_BCs import *
from feniQS.structure.helper_loadings import *
from feniQS.problem.problem_plastic import *
from feniQS.material.constitutive_plastic import *
from feniQS.problem.time_stepper import *
from feniQS.problem.post_process import *

class FenicsConfig:
    el_family = 'Lagrange'
    shF_degree_u = 1
    shF_degree_ebar = 1
    integ_degree = 2
    
class ParsBeam2dPlastic:
    def __init__(self):
        self.constraint = 'PLANE_STRESS'
        
        self.E = 70e3
        self.nu = 0.3
        self.lx = 100
        self.ly = 10
        
        self.x_from = 17 * self.lx / 40
        self.x_to = 23 * self.lx / 40
        
        self.unit_res = 0.4
        self.refined_s = 2
        
        self.loading = ParsLoading(level=self.ly / 10)
        self.loading.scales = np.array([1])
        self.loading.N = 1.25
        
        # yield strength
        self.sig0 = 150.0
        
        # Hardening
        Et = self.E / 100.0
        self.hardening_isotropic_law = {
            'law': 'linear',
            'modulus': 10 * self.E * Et / (self.E - Et),
            'sig_u': None, # No ultimate strength
            }
        
        self._plot = True
        self._write_files = True
        
        self.f = df.Constant((0.0, 0.0))

class Bend3points2dPlasticModel:
    def __init__(self, pars, _name=None):
        self.pars = pars
        if _name is None:
            self._name = 'Bend3points2dPlasticModel'
        self._set_logger_and_path()
        self._soften_pars()
        self._establish_model()
        
    def _set_logger_and_path(self):
        # Setting name and path for results
        self._path = f"./examples/fem/models_QS/{self._name}/"
        make_path(self._path)
        self.logger = LoggerSetup(self._path + self._name + '.log')
        
    def _soften_pars(self):
        pass # some issues emerge when using dolfin functions for parameters like E
        ## WAY 1: using varaible
        # self.pars.E = df.variable(self.pars.E)
        
        ## WAY 2: using constant
        # self.pars.E = df.Constant(self.pars.E)
    
    def _establish_model(self):
        ### MESH ###
        # mesh = df.RectangleMesh(df.Point(0,0), df.Point(lx,ly), 10, 10)
        mesh = refined_rectangle_mesh(self.pars.lx, self.pars.ly, self.pars.x_from, self.pars.x_to \
                                      , unit_res=self.pars.unit_res, refined_s=self.pars.refined_s)
        
        ### MATERIAL ###
        yf = Yield_VM(self.pars.sig0, constraint=self.pars.constraint
                      , hardening_isotropic_law=self.pars.hardening_isotropic_law)
        ## perfect plasticity (No hardening)
        # mat = PlasticConsitutivePerfect(self.pars.E, nu=self.pars.nu \
        #                                 , constraint=self.pars.constraint, yf=yf)
        ## Isotropic-hardenning plasticity
        ri = RateIndependentHistory() # p = 1, i.e.: kappa_dot = lamda_dot
        # ri = RateIndependentHistory_PlasticWork(yf) # kappa_dot = plastic work rate
        mat = PlasticConsitutiveRateIndependentHistory(self.pars.E, nu=self.pars.nu \
                                        , constraint=self.pars.constraint, yf=yf, ri=ri)
        
        #### PROBLEM ###
        self.fen = FenicsPlastic(mat, mesh, fen_config=FenicsConfig)
        self.fen.build_variational_functionals(f=self.pars.f)
        
        #### BCs and LOADs ###
        self.load_expr = time_varying_loading(fs=self.pars.loading.level*self.pars.loading.scales, N=self.pars.loading.N, T=self.pars.loading.T, _case=self.pars.loading.case \
                                              , _path=self._path)
        bcs_DR, bcs_DR_inhom = load_and_bcs_on_3point_bending( \
                                mesh, self.pars.lx, self.pars.ly, self.pars.x_from, self.pars.x_to, self.fen.i_u, self.load_expr)
        self.fen.bcs_DR = []; self.fen.bcs_DR_dofs = []
        for bc in bcs_DR.values():
            self.fen.bcs_DR.append(bc['bc'])
            self.fen.bcs_DR_dofs.extend(bc['bc_dofs'])
        self.fen.bcs_DR_inhom = []; self.fen.bcs_DR_inhom_dofs = []
        for bc in bcs_DR_inhom.values():
            self.fen.bcs_DR_inhom.append(bc['bc'])
            self.fen.bcs_DR_inhom_dofs.extend(bc['bc_dofs'])
        
        ## BUILD SOLVER ###
        self.fen.build_solver(time_varying_loadings=[self.load_expr])
        
    def solve(self, t_end, checkpoints=[], t_start=0, dt=None, reaction_dofs=[], \
              time_adjusting_methods=[], time_adjusting_args={}, other_pps=[], _reset=True):
        
        pps = []
        pps.append(PostProcessPlastic(fen=self.fen, _name=self._name, out_path=self._path \
                                      , reaction_dofs=reaction_dofs, write_files=self.pars._write_files))
        pps += other_pps
        
        # define time-stepper
        ts = TimeStepper(self.fen.solve, pps, \
                         time_adjusting_methods=time_adjusting_methods, time_adjusting_args=time_adjusting_args, \
                         logger=self.logger.file, solution_field=self.fen.get_F_and_u()[1], \
                         increase_num_iter=8, increase_factor=1.2, _dt_max=0.1)
        # solve
        t_res = 31
        if dt==None:
            dt = t_end / t_res
            
        # iterations = ts.adaptive(t_end=t_end, t_start=t_start, dt=dt, checkpoints=checkpoints)
        iterations = ts.equidistant(1, dt=dt)
        
        if self.pars._plot:
            fig = plt.figure()
            plt.plot(ts.ts, iterations, marker='.')
            plt.title('Number of iterations over time\nSum = ' + str(sum(iterations)) + ', # time steps = ' + str(len(ts.ts)))
            plt.xlabel('t')
            plt.ylabel('Iterations number')
            plt.savefig(self._path + 'iterations.pdf')
            
            pps[0].plot_reaction_forces(tits=['Reaction force at the middle'], full_file_names=[self._path + 'reaction_force.pdf'])
            
            pps[0].plot_residual_at_free_DOFs('Residual-norm of free DOFs', full_file_name=self._path + 'residual_norm_free_DOFs.pdf')
            plt.show()
        
        print('One FEniCS problem of "' + self._name + '" was solved.')
        return pps
    
if __name__ == "__main__":
    df.set_log_level(30)
    
    pars = ParsBeam2dPlastic()
    
    model = Bend3points2dPlasticModel(pars)
    checkpoints = []
    t_end = model.pars.loading.T
    reaction_dofs = [model.fen.bcs_DR_inhom_dofs]
        
    pps = model.solve(t_end, reaction_dofs=reaction_dofs, checkpoints=checkpoints)
    
    ## print out minimum and maximum reaction forces
    ff = [sum(ri) for ri in pps[0].reaction_forces[0]]
    print(f"F_min, F_max = {min(ff)}, {max(ff)} .")
    
    df.parameters["form_compiler"]["representation"] = "uflacs" # back to default setup of dolfin