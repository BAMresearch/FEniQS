"""
QSM:
    Quasi Static Model
GDM:
    Gradient Damage Material Model
"""
from feniQS.problem.model_time_varying import *
from feniQS.problem.post_process import *
from feniQS.general.yaml_functions import *

pth_QSM_GDM = CollectPaths('QSM_GDM.py')
pth_QSM_GDM.add_script(pth_model_time_varying)
pth_QSM_GDM.add_script(pth_post_process)
pth_QSM_GDM.add_script(pth_yaml_functions)

def get_QSM_GDM(pars_struct, cls_struct, pars_gdm, _path=None, _name=None):
    struct = cls_struct(pars=pars_struct, _path=_path, _name=_name)
    return QSModelGDM(pars=pars_gdm, struct=struct, _path=_path, _name=_name)

class QSModelGDM(QuasiStaticModel):
    def __init__(self, pars, struct, _path=None, _name=None \
                 , penalty_dofs=[], penalty_weight=0.):
        if isinstance(pars, dict):
            pars = ParsBase(**pars)
        ## The parameters of model might include structure parameters, thus, it must be updated accordingly.
        if struct.pars is not None:
            for kp in pars.__dict__.keys():
                if kp in struct.pars.__dict__.keys():
                    pars.__dict__[kp] = struct.pars.__dict__[kp]
        if _name is None:
            _name = 'QsGdm_' + struct._name
        self.struct = struct
        if not pars.mat_type.lower()=='gdm':
            raise ValueError('The material type must be GDM (gradient damage model).')
        self.penalty_dofs = penalty_dofs
        self.penalty_weight = df.Constant(penalty_weight)
        QuasiStaticModel.__init__(self, pars, pars.softenned_pars, _path, _name)
    
    def establish_model(self): # over-written
        if self.pars._write_files:
            _n = 'model' if self._name=='' else self._name
            write_to_xdmf(self.struct.mesh, xdmf_name=_n+'_mesh.xdmf', xdmf_path=self._path)
        
        ### MATERIAL ###
        if self.pars.damage_law == 'perfect':
            gK = GKLocalDamagePerfect(K0=self.pars.e0 + self.pars.e0_min)
        elif self.pars.damage_law=='exp':
            gK = GKLocalDamageExponential(e0=self.pars.e0+self.pars.e0_min \
                                          , ef=self.pars.ef+self.pars.ef_min, alpha=self.pars.alpha)
        else:
            raise NotImplementedError('The damage law is not considered.')
        mat = GradientDamageConstitutive(E=self.pars.E + self.pars.E_min, nu=self.pars.nu \
                                        , constraint=self.pars.constraint, gK=gK \
                                        , c_min=self.pars.c_min, c=self.pars.c)
        
        ### PROBLEM ###
        self.fen = FenicsGradientDamage(mat, self.struct.mesh, fen_config=self.pars, jac_prep=self.pars.analytical_jac \
                                        , dep_dim=self.struct.mesh.geometric_dimension() \
                                        , penalty_dofs=self.penalty_dofs, penalty_weight=self.penalty_weight)
        # build variationals
        try:
            sigma_scale = self.struct.special_fenics_fields['sigma_scale']
        except KeyError:
            sigma_scale = 1.0
        self.fen.bcs_NM_tractions, self.fen.bcs_NM_measures = self.struct.get_tractions_and_dolfin_measures()
        self.fen.concentrated_forces = self.struct.get_concentrated_forces()
        self.time_varying_loadings = {}
        for direction, cfs in self.fen.concentrated_forces.items():
            for i, cf in enumerate(cfs):
                ff = cf['value']
                if (isinstance(ff, df.Expression) or isinstance(ff, df.UserExpression)) and hasattr(ff, 't'):
                    self.time_varying_loadings.update({f"f_concentrated_{direction}_{i}": ff})
        self.fen.build_variational_functionals(f=self.pars.f, integ_degree=self.pars.integ_degree, expr_sigma_scale=sigma_scale)
        self.fen.bcs_NM_dofs = self.struct.get_tractions_dofs(i_full=self.fen.get_i_full(), i_u=self.fen.get_iu())
        
        ### POST-PROCESS (default: regardless of solve_options) ###
        self.pps_default = {} # By default, no postprocessor (e.g. to compute/store residuals, for analytical computation of Jacobian)
        
        self.established = True
        
        self._commit_struct_to_model()
    
    def build_solver(self, solve_options):
        ### BUILD SOLVER ###
        tvls = list(self.time_varying_loadings.values()) # The order does not matter for adjusting 't' of time_varying_loadings
        self.fen.build_solver(solver_options=solve_options.solver_options, time_varying_loadings=tvls)
        self.solver_built = True
        
    def set_pps(self, solve_options, DG_degree=1):
        self.pps = []
        reaction_dofs = self.get_reaction_dofs(reaction_places=solve_options.reaction_places)
        _n = 'model' if self._name=='' else self._name
        self.pps.append(PostProcessGradientDamage(self.fen, _n, self._path \
                        , reaction_dofs=reaction_dofs, write_files=self.pars._write_files, DG_degree=DG_degree))
        for pp in self.pps_default.values():
            self.pps.append(pp)
    
    def solve(self, solve_options, other_pps=[], _reset_pps=True):
        ts, iterations, solve_options = QuasiStaticModel.solve(self, solve_options, other_pps, _reset_pps)
        return ts, iterations
    
    def _commit_struct_to_model(self):
        """
        To commit BCs and time-varying loadings from structure to the model.
        """
        bcs_hom, bcs_inhom, time_varying_loadings = self.struct.get_BCs(self.fen.get_iu())
        self.time_varying_loadings.update(time_varying_loadings)
        bcs_hom = [bc['bc'] for bc in bcs_hom.values()]
        bcs_inhom = [bc['bc'] for bc in bcs_inhom.values()]
        self.revise_BCs(remove=True, new_BCs=bcs_hom, _as='hom')
        self.revise_BCs(remove=False, new_BCs=bcs_inhom, _as='inhom')
    
    def get_reaction_dofs(self, reaction_places):
        return self.struct.get_reaction_dofs(reaction_places, i_u=self.fen.get_iu())

    