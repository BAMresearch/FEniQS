from feniQS.problem.model_time_varying import *
from feniQS.problem.post_process import *
from feniQS.general.yaml_functions import *

pth_QSM_Elastic = CollectPaths('./feniQS/problem/QSM_Elastic.py')
pth_QSM_Elastic.add_script(pth_model_time_varying)
pth_QSM_Elastic.add_script(pth_post_process)
pth_QSM_Elastic.add_script(pth_yaml_functions)

def get_QSM_Elastic(pars_struct, cls_struct, pars_elastic, _path=None, _name=None):
    struct = cls_struct(pars_struct, _path=_path, _name=_name)
    return QSModelElastic(pars=pars_elastic, struct=struct, _path=_path, _name=_name)

class QSModelElastic(QuasiStaticModel):
    def __init__(self, pars, struct, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsBase(**pars)
        ## The parameters of model might include structure parameters, thus, it must be updated accordingly.
        if struct.pars is not None:        
            for kp in pars.__dict__.keys():
                if kp in struct.pars.__dict__.keys():
                    pars.__dict__[kp] = struct.pars.__dict__[kp]
        if _name is None:
            _name = 'QsElastic_' + struct._name
        self.struct = struct
        if not pars.mat_type.lower()=='elastic':
            raise ValueError('The material type must be elastic.')
        QuasiStaticModel.__init__(self, pars, pars.softenned_pars, _path, _name)
    
    def establish_model(self): # over-written
        if self.pars._write_files:
            write_to_xdmf(self.struct.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
        
        ### MATERIAL ###
        mat = ElasticConstitutive(E=self.pars.E + self.pars.E_min, nu=self.pars.nu, constraint=self.pars.constraint)
        
        ### PROBLEM ###
        self.fen = FenicsElastic(mat, self.struct.mesh, fen_config=self.pars)
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
        
    def set_pps(self, solve_options, DG_degree=0):
        self.pps = []
        reaction_dofs = self.get_reaction_dofs(reaction_places=solve_options.reaction_places)
        self.pps.append(PostProcessElastic(self.fen, self._name, self._path \
                        , reaction_dofs=reaction_dofs, write_files=self.pars._write_files))
        for pp in self.pps_default.values():
            self.pps.append(pp)
    
    def solve(self, solve_options, other_pps=[], _reset_pps=True, write_pars=True):
        ts, iterations, solve_options = QuasiStaticModel.solve(self, solve_options=solve_options \
                                        , other_pps=other_pps, _reset_pps=_reset_pps, write_pars=write_pars)
        return ts, iterations
    
    def _commit_struct_to_model(self):
        """
        To commit BCs and time-varying loadings from structure to the model.
        Also, to handle penalizing DOFs; i.e. putting linear springs at certain DOFs.
        """
        bcs_hom, bcs_inhom = self.struct.get_BCs(i_u=self.fen.get_iu(), fresh=True)
        self.time_varying_loadings.update(self.struct.get_time_varying_loadings())
        bcs_hom_bcs = [bc['bc'] for bc in bcs_hom.values()]
        bcs_hom_dofs = [bc['bc_dofs'] for bc in bcs_hom.values()]
        bcs_inhom_bcs = [bc['bc'] for bc in bcs_inhom.values()]
        bcs_inhom_dofs = [bc['bc_dofs'] for bc in bcs_inhom.values()]
        self.revise_BCs(remove=True, new_BCs=bcs_hom_bcs, new_BCs_dofs=bcs_hom_dofs, _as='hom')
        self.revise_BCs(remove=False, new_BCs=bcs_inhom_bcs, new_BCs_dofs=bcs_inhom_dofs, _as='inhom')
        
        penalty_features = self.struct.get_penalty_features(self.fen.get_iu())
        if len(penalty_features)>0:
            pws = [] # penalty weights
            pdofss = []
            for penalty in penalty_features.values():
                pw = penalty['weight']
                if penalty['u0']!=0. or not any([isinstance(pw, a) for a in [int, float, np.float64]]):
                    raise NotImplementedError(f"Penalty feature is not implemented for nonzero reference disp. and for non-uniform penalty weight.")
                pw0 = pw if len(pws)==0 else pws[-1]
                if pw!=pw0:
                    raise NotImplementedError(f"Penalty feature is not implemented for multiple different penalty weights.")
                pws.append(pw)
                pdofss += penalty['dofs']
            self.penalty_dofs = pdofss
            self.penalty_weight.assign(pw) # the same
    
    def get_reaction_dofs(self, reaction_places):
        return self.struct.get_reaction_dofs(reaction_places, i_u=self.fen.get_iu())

def run_QSM_elastic_default(pars, solve_options, cls_struct \
                                  , _name=None, _path=None, _msg=True, _return=True):
    ### MODEL ###
    model = get_QSM_Elastic(
        pars_struct=pars,
        cls_struct=cls_struct,
        pars_elastic=pars,
        _path=_path,
        _name=_name,
    )
    ### SOLVE ###
    model.solve(solve_options)
    ### POST-PROCESS ###
    pp0 = model.pps[0]
    ## Reaction forces
    try:
        rps = solve_options.reaction_places
    except AttributeError:
        rps = solve_options['reaction_places']
    tits = [f"Reaction force at {rp}" for rp in rps]
    file_names = [f"{model._path}reaction_force_{rp}" for rp in rps]
    fss = pp0.plot_reaction_forces(tits=tits, full_file_names=file_names)
    try:
        struct_loadings = model.struct.loadings
        for ip, rp in enumerate(rps):
            try:
                struct_loading = struct_loadings[rp]
                loaded_us = evaluate_expression_of_t(struct_loading, ts=pp0.ts)[1]
                fs = fss[ip]
                fig = plt.figure()
                plt.plot(loaded_us, fs, marker='*', linestyle='--')
                plt.title(f"Load-displacement at {rp}")
                plt.xlabel('Displacement')
                plt.ylabel('Force')
                file_name = f"{model._path}load_displacement_{rp}"
                plt.savefig(f"{file_name}.png", bbox_inches='tight', dpi=500)
                plt.show()
                ld = np.array([loaded_us, fs]).T
                yamlDump_array(ld, f"{file_name}.yaml")
            except KeyError:
                print(f"WARNING (Post-process):\n\tThe structure's loadings attribute does not include reaction_place={rp}. Respective load-displacement data was not extracted.")
    except AttributeError:
        print(f"WARNING (Post-process):\n\tThe structure has no attribute 'loadings' containing dictionary of loading signals. No load-displacement data was extracted.")
    ## Stresses
    try:
        sps = solve_options.stress_places
    except AttributeError:
        sps = solve_options['stress_places']
    try:
        stress_points = model.struct.get_coordinates(coordinate_places=sps, resolution=None)
        for sp in sps:
            _points = stress_points[sp]
            if (_points is None) or (len(_points)==0):
                print(f"WARNING (Post-process):\n\tStructure's 'get_coordinates' method returned no points for stress_place '{sp}'. Respective stress data was not extracted.")
            else:
                _points = np.array(_points)
                _stresses = pp0.eval_checked_sig(points=_points)
                _sig_comp = np.array(_stresses['components']) # hierarchy : time, points, stress_components
                _sig_VM = np.array(_stresses['VM']) # hierarchy : time, points
                yamlDump_array(_points, f"{model._path}coordinates_{sp}.yaml")
                yamlDump_array(_sig_comp, f"{model._path}stresses_components_{sp}.yaml")
                yamlDump_array(_sig_VM, f"{model._path}stresses_VM_{sp}.yaml")
    except (NotImplementedError):
        print(f"WARNING (Post-process):\n\tAll stress_places '{sps}' were ignored. The 'get_coordinates' method is not implemented/overwritten for the respective structure.")
    ### RETURNs ###
    if _msg:
        print(f"Quasi-static elastic results stored in:\n\t'{model._path}'.")
    if _return:
        return model