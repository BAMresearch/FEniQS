from feniQS.structure.helper_BCs import *
from feniQS.structure.helper_loadings import *
from feniQS.problem.problem_plastic import *
from feniQS.material.constitutive_plastic import *
from feniQS.problem.time_stepper import *
from feniQS.problem.post_process import *

_format = '.png'
# _format = '.pdf'

class FenicsConfig:
    el_family = 'Lagrange'
    shF_degree_u = 3
    integ_degree = 4

def read_ref_sol_MATLAB(_file = './examples/fem/models_QS/cantBeamPlasticRefSol2d_MATLAB/Results.mat'):
    from scipy.io import loadmat
    _data = loadmat(_file)
    u = _data['results'][0][0][0][0]
    f = _data['results'][0][0][1][0]
    return u, -f

class ParsCantileverBeam:
    def __init__(self):
        self.constraint = 'PLANE_STRESS'
        # self.constraint = '3D'
        
        self.E = 1000.
        self.nu = 0.3
        self.lx = 6.0
        self.ly = 0.5
        self.lz = 0.5 # only relevant for 3-D case or ploting reaction force in 2-D case
        
        self.unit_res = 6
        
        self.loading = ParsLoading(level=4*self.ly)
        self.loading.scales = np.array([1])
        
        self.sol_tol = 1e-8
        if self.loading.case == 'ramp':
            self.sol_res = 61
        else:
            self.sol_res = int(self.loading.N * 83)
        
        # body force (zero)
        if self.constraint=='PLANE_STRESS' or self.constraint=='PLANE_STRAIN':
            self.geo_dim = 2
        elif self.constraint=='3D':
            self.geo_dim = 3
        self.f = df.Constant(self.geo_dim * [0.0])
        
        # others
        self._plot = True
        self._write_files = True

class ParsCantileverBeamPlastic(ParsCantileverBeam):
    def __init__(self):
        super().__init__()
        
        # yield strength
        self.sig0 = 12.0
        # self.sig0 = 1e13 # leads to elastic case
        
        ### NO Hardening
        # self.H = 0.0 # Hardening modulus
        
        ### ISOTROPIC Hardening
        Et = self.E / 100.0
        self.H = 15 * self.E * Et / (self.E - Et) # Hardening modulus
        ## Hardening hypothesis
        self.hardening_hypothesis = 'unit' # denoting the type of the harding function "P(sigma, kappa)"
        # self.hardening_hypothesis = 'plastic-work'

class CantileverBeamPlasticModel:
    def __init__(self, pars, _name=None):
        self.pars = pars
        if _name is None:
            _name = 'CantBeamPlastic_deg' + str(FenicsConfig.shF_degree_u) + '_' + str(self.pars.constraint) + '_H=' \
                + '%.1f'%self.pars.H
            if self.pars.H != 0:
                _name  += '_P=' + self.pars.hardening_hypothesis
        self._name = _name
        self._set_logger_and_path()
        self._soften_pars()
        self._establish_model()
        
    def _set_logger_and_path(self):
        # Setting name and path for results
        self._path = str(Path(__file__).parent) + '/' + self._name + '/'
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
        res_x = int(self.pars.unit_res * self.pars.lx)
        res_y = int(self.pars.unit_res * self.pars.ly)
        if self.pars.geo_dim==2:
            mesh = df.RectangleMesh(df.Point(0,0), df.Point(self.pars.lx, self.pars.ly), res_x, res_y)
        elif self.pars.geo_dim==3:
            res_z = int(self.pars.unit_res * self.pars.lz)
            mesh = df.BoxMesh(df.Point(0.0, 0.0, 0.0), df.Point(self.pars.lx, self.pars.ly, self.pars.lz) \
                              , res_x, res_y, res_z)
        
        ### MATERIAL ###
        yf = Yield_VM(self.pars.sig0, constraint=self.pars.constraint, H=self.pars.H)
        if self.pars.H == 0: ## perfect plasticity (No hardening)
            mat = PlasticConsitutivePerfect(self.pars.E, nu=self.pars.nu, constraint=self.pars.constraint, yf=yf)
        else: ## Isotropic-hardenning plasticity
            if self.pars.hardening_hypothesis == 'unit':
                ri = RateIndependentHistory() # p = 1, i.e.: kappa_dot = lamda_dot
            elif self.pars.hardening_hypothesis == 'plastic-work':
                ri = RateIndependentHistory_PlasticWork(yf) # kappa_dot = plastic work rate
            mat = PlasticConsitutiveRateIndependentHistory(self.pars.E, nu=self.pars.nu, constraint=self.pars.constraint, yf=yf, ri=ri)
        
        #### PROBLEM ###
        self.fen = FenicsPlastic(mat, mesh, fen_config=FenicsConfig)
        self.fen.build_variational_functionals(f=self.pars.f)
        
        #### BCs and LOADs ###
        self.load_expr = time_varying_loading(fs=self.pars.loading.level*self.pars.loading.scales, N=self.pars.loading.N, T=self.pars.loading.T, _case=self.pars.loading.case \
                                              , _path=self._path, _format=_format)
        if self.pars.geo_dim == 2:
            self.fen.bcs_DR \
                , self.fen.bcs_DR_dofs \
                    , self.fen.bcs_DR_inhom \
                        , self.fen.bcs_DR_inhom_dofs \
            = load_and_bcs_on_cantileverBeam2d(mesh, self.pars.lx, self.pars.ly, self.fen.i_u, self.load_expr)
        elif self.pars.geo_dim == 3:
            self.fen.bcs_DR \
                , self.fen.bcs_DR_dofs \
                    , self.fen.bcs_DR_inhom \
                        , self.fen.bcs_DR_inhom_dofs \
            = load_and_bcs_on_cantileverBeam3d(mesh, self.pars.lx, self.pars.ly, self.pars.lz, self.fen.i_u, self.load_expr)
        
        #### BUILD SOLVER ###
        self.fen.build_solver(time_varying_loadings=[self.load_expr], tol=self.pars.sol_tol)
    
    def solve(self, t_end, checkpoints=[], t_start=0, dt=None, reaction_dofs=[], \
              time_adjusting_methods=[], time_adjusting_args={}, other_pps=[], _reset=True):
        
        pps = []
        pps.append(PostProcessPlastic(self.fen, self._name, self._path, reaction_dofs=reaction_dofs, write_files=self.pars._write_files))
        pps += other_pps
        
        # define time-stepper
        ts = TimeStepper(self.fen.solve, pps, \
                         time_adjusting_methods=time_adjusting_methods, time_adjusting_args=time_adjusting_args, \
                         logger=self.logger.file, solution_field=self.fen.get_F_and_u()[1], \
                         increase_num_iter=8, increase_factor=1.2, _dt_max=0.1)
        # solve
        if dt==None:
            dt = t_end / self.pars.sol_res
            
        # iterations = ts.adaptive(t_end=t_end, t_start=t_start, dt=dt, checkpoints=checkpoints)
        iterations = ts.equidistant(t_end, dt=dt)
        
        if self.pars._plot:
            fig = plt.figure()
            plt.plot(ts.ts, iterations, marker='.')
            plt.title('Number of iterations over time\nSum = ' + str(sum(iterations)) + ', # time steps = ' + str(len(ts.ts)))
            plt.xlabel('t')
            plt.ylabel('Iterations number')
            plt.savefig(self._path + 'iterations' + _format)
            
            if len(reaction_dofs) > 0:
                sz=14
                _tit = 'Reaction force at top-left node, ' + str(self.pars.constraint) + ', H=' + '%.1f'%self.pars.H
                if self.pars.H != 0:
                    _tit  += ', P=' + self.pars.hardening_hypothesis
                
                if self.pars.geo_dim == 2:
                    _factor = self.pars.lz
                elif self.pars.geo_dim == 3:
                    _factor = 1.0
                pps[0].plot_reaction_forces(tits=[_tit], full_file_names=[self._path + self._name + '_reaction_force' + _format], factor=_factor)
                
                ## together with the linear tangent stiffness and possible reference MATLAB solution
                _, u_sol = evaluate_expression_of_t(self.load_expr, t0=0.0, t_end=t_end, _res=self.pars.sol_res)
                plt.figure()
                if self.pars.geo_dim == 2: # to plot reference MATLAB solution
                    f_sol = self.pars.lz * np.array(pps[0].reaction_forces[0])
                    if self.pars.H == 0 and self.pars.loading.level == 0.65 and self.pars.loading.case=='ramp':
                        _u, _f = read_ref_sol_MATLAB()
                        plt.plot(-_u, self.pars.lz * _f, marker='', label='MATLAB')
                elif self.pars.geo_dim == 3:
                    f_sol = [sum(f) for f in pps[0].reaction_forces[0]]
                plt.plot(u_sol, f_sol, marker='.', label='FEniCS: Plasticity')
                K_analytical = self.pars.lz * (self.pars.ly ** 3) * self.pars.E / 4 / (self.pars.lx ** 3)
                u_analytical = [u for u in u_sol if (K_analytical*u<=max(f_sol) and K_analytical*u>=min(f_sol))]
                f_analytical = K_analytical * np.array(u_analytical)
                plt.plot(u_analytical, f_analytical, label='Equivalent elastic stiffness')
                plt.title(_tit, fontsize=sz)
                plt.xlabel('u', fontsize=sz)
                plt.ylabel('f', fontsize=sz)
                plt.xticks(fontsize=sz)
                plt.yticks(fontsize=sz)
                plt.legend()
                plt.savefig(self._path + self._name + '_reaction_force_compare' + _format)
                plt.show()
            
            pps[0].plot_residual_at_free_DOFs('Residual-norm of free DOFs', full_file_name=self._path + self._name + '_residual_norm_free_DOFs' + _format)
            plt.show()
            
            write_attributes(self.pars, _name=self._name, _path=self._path)
        
        print('One FEniCS problem of "' + self._name + '" was solved.')
        return pps, [[u_analytical, u_sol], [f_analytical, f_sol], ['Equivalent elastic stiffness', 'FEniCS: Plasticity']]
    
if __name__ == "__main__":
    df.set_log_level(30)
    
    pars = ParsCantileverBeamPlastic()
    
    model = CantileverBeamPlasticModel(pars)
    checkpoints = []
    reaction_dofs = [model.fen.bcs_DR_inhom_dofs]
    aa = read_ref_sol_MATLAB()
    # pps, plots_data = model.solve(pars.loading.t_end, reaction_dofs=reaction_dofs, checkpoints=checkpoints)
    
    ## print out minimum and maximum reaction forces
    ff = [sum(ri) for ri in pps[0].reaction_forces[0]]
    msg = f"F_min, F_max = {min(ff)}, {max(ff)} ."
    model.logger.file.debug('\n' + msg)
    print(msg)
    
    df.parameters["form_compiler"]["representation"] = "uflacs" # back to default setup of dolfin
    

