import dolfin as df
import matplotlib.pyplot as plt
from feniQS.general.general import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.problem.problem import *

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid") # style must be one of: white, dark, whitegrid, darkgrid, ticks
except ModuleNotFoundError:
    print(f"\n\n\t{'-' * 70}\n\tWARNING: It is recommended to install 'seaborn' to get nicer plots.\n\t{'-' * 70}\n\n")

pth_post_process = CollectPaths('post_process.py')
pth_post_process.add_script(pth_general)
pth_post_process.add_script(pth_fenics_functions)
pth_post_process.add_script(pth_problem)

def compute_Dg_dp(dg_dp, dg_du, dg_dK, Du_dp, DK_dp, free_dofs):
    # based on: https://git.bam.de/mechanics/ajafari/texts/-/blob/txt03_fw_2_incrementally/03_jacobian_by_perturbation/jac_of_fw_perturbation.pdf
    return dg_dp + dg_du[:, free_dofs] @ Du_dp + dg_dK @ DK_dp
    
class PostProcessEvalForm:
    def __init__(self, form, dofs=None, checkpoints=[]):
        """
        This calss provides a callable for evaluating a FEniCS form and storing it at some checkpoints (times).
        """
        self.form = form
        self.dofs = dofs
        self.checked = []
        self.checkpoints = checkpoints # will be overwritten as soon as a time-stepper incremental solution is executed.
    
    def __call__(self, t, logger=None):
        if t==0.0:
             self.checked = []
        f = df.assemble(self.form)
        if isinstance(f, df.Vector):
            f = f.get_local()
            if self.dofs is not None:
                f = f[self.dofs]
        elif isinstance(f, df.Matrix):
            f = f.array()
            if self.dofs is not None:
                f = f[np.ix_(self.dofs, self.dofs)]
        else:
            raise TypeError(f"The type of the assembled form is not recognized.")
        for tt in self.checkpoints:
            if abs(t - tt) < 1e-9:
                self.checked.append(f)
        return f
    
class PostProcessEvalFunc:
    def __init__(self, func, dofs=None, checkpoints=[]):
        """
        This calss provides a callable for getting values (vector) of a FEniCS function and appending it to an array over time.
        """
        self.func = func
        self.all = []
        self.checked = []
        self.dofs = dofs
        self.checkpoints = checkpoints # will be overwritten as soon as a time-stepper incremental solution is executed.
    
    def __call__(self, t, logger=None):
        if t==0.0: # initiation
            self.all = []
            self.checked = []
        f = self.func.vector().get_local()
        if self.dofs is not None:
            f = f[self.dofs]
        self.all.append(f)
        if self.checkpoints!=[]:
            for tt in self.checkpoints:
                if abs(t - tt) < 1e-9:
                    self.checked.append(f)
        return f

class PostProcess:
    def __init__(self, fen, _name='', out_path=None, reaction_dofs=[], log_residual_vector=False, write_files=True):
        self.fen = fen
        self.reaction_dofs = reaction_dofs # for which, the reaction force (residual) will be calculated
        self.log_residual_vector = log_residual_vector
        
        self.ts = [] # list of times
        self.checkpoints = [] # will be assigned as soon as a time-stepper incremental solution is executed.
        
        self.reaction_forces = []
        self.reaction_forces_checked = []
        for egal in range(len(self.reaction_dofs)):
            self.reaction_forces.append([])
            self.reaction_forces_checked.append([])
            # Every entry is pertaining to a group of DOFs (e.g. a reaction place) and a list itslef
            # Every entry of that list is for a certain time ("t") and again a list of values over the corresponding DOFs.
            # So, the hierarchy is: reactions-group, time, DOFs
        self.residual_norms = [] # # Every entry is for a certain time ("t") and is the residual norm at all FREE DOFs (excluding Dirichlet BCs)
        
        self.write_files = write_files
        if out_path is None:
            out_path = './'
        make_path(out_path)
        self.full_name = out_path + _name
        self.xdmf_name = self.full_name
        self.xdmf_checked_name = self.full_name + '_checked' # excluding 'xdmf' or 'h5' extensions.
        
        self.remove_files()
        
        self.xdmf_checked = df.XDMFFile(self.xdmf_checked_name + '.xdmf') # This stores the full Functions with function-space data.
        
        if self.write_files:
            self.R = df.Function(self.fen.get_i_full(), name='Residual')
            self.xdmf = df.XDMFFile(self.xdmf_name + '.xdmf') # This is just for visualization
            self.xdmf.parameters["functions_share_mesh"] = True
            self.xdmf.parameters["flush_output"] = True
        
    def __call__(self, t, logger):
        self.ts.append(t)
        
        reaction_forces_groups, b_norm = compute_residual(F=self.fen.get_F_and_u()[0], bcs_dofs=self.fen.bcs_DR_dofs + self.fen.bcs_DR_inhom_dofs, \
                                             reaction_dofs=self.reaction_dofs, logger=logger, write_residual_vector=self.log_residual_vector)
        assert len(reaction_forces_groups) == len(self.reaction_dofs)
        self.residual_norms.append(b_norm)
        
        u_plot = self.fen.get_uu()
        for tt in self.checkpoints:
            if abs(t - tt) < 1e-9:
                self.xdmf_checked.write_checkpoint(u_plot, 'u', t, append=True)
                for ii, ff in enumerate(reaction_forces_groups):
                    self.reaction_forces_checked[ii].append(ff) # hierarchy: reactions-group, time, DOFs
                break
        for ii, ff in enumerate(reaction_forces_groups):
            self.reaction_forces[ii].append(ff) # hierarchy: reactions-group, time, DOFs
        
        if self.write_files:
            self.xdmf.write(u_plot, t)
            self.R.vector().set_local(df.assemble(self.fen.get_F_and_u()[0]))
            self.xdmf.write(self.R, t)
    
    def plot_reaction_forces(self, tits, full_file_names=None, dof='sum', factor=1, marker='.', sz=14):
        Fs = []
        for ii in range(len(self.reaction_forces)):
            fs = self.reaction_forces[ii]
            if dof=='sum':
                if type(fs[0])==list or type(fs[0])==np.ndarray:
                    f_dof = [factor * sum(f) for f in fs]
                else:
                    f_dof = [factor * f for f in fs]
            else:
                f_dof = [factor * f[dof] for f in fs]
            fig1 = plt.figure()
            plt.plot(self.ts, f_dof, marker=marker)
            plt.title(tits[ii], fontsize=sz)
            plt.xlabel('t', fontsize=sz)
            plt.ylabel('f', fontsize=sz)
            plt.xticks(fontsize=sz)
            plt.yticks(fontsize=sz)
            if full_file_names is not None:
                plt.savefig(full_file_names[ii], bbox_inches='tight', dpi=300)
            plt.show()
            Fs.append(f_dof)
        return Fs
        
    def plot_residual_at_free_DOFs(self, tit, full_file_name=None, factor=1, marker='.'):
        fig1 = plt.figure()
        plt.plot(self.ts, [factor * res for res in self.residual_norms], marker=marker)
        plt.title(tit)
        plt.xlabel('t')
        plt.ylabel('Residual')
        if full_file_name is not None:
            plt.savefig(full_file_name, bbox_inches='tight', dpi=300)
        plt.show()
    
    def eval_checked_u(self, points):
        v = self.fen.get_iu(_collapse=True)
        u_read = df.Function(v)
        vals = []
        num_checkpoints = len(self.checkpoints)
        for ts in range(num_checkpoints):
            self.xdmf_checked.read_checkpoint(u_read, 'u', ts)
            u_ts = [u_read(p) for p in points]
            vals.append(np.array(u_ts))
        return vals
    
    def eval_checked_reaction_forces(self):
        return self.reaction_forces_checked # hierarchy : reactions-group, time, DOFs
    
    def close_files(self):
        self.xdmf_checked.close()
        if self.write_files:
            self.xdmf.close()
    
    def remove_files(self):
        import os
        fs = [self.xdmf_checked_name + '.xdmf', self.xdmf_checked_name + '.h5']
        if self.write_files:
            fs += [self.xdmf_name + '.xdmf', self.xdmf_name + '.h5']
        for f in fs:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        for ii in range(len(self.reaction_dofs)):
            _ff = self.full_name  + '_reaction_forces_' + str(ii) + '.h5'
            try:
                os.remove(_ff)
            except FileNotFoundError:
                pass

class PostProcessGradientDamage(PostProcess):
    def __init__(self, fen, _name='', out_path=None, reaction_dofs=None, log_residual_vector=False, write_files=True, DG_degree=1):
        super().__init__(fen, _name, out_path, reaction_dofs, log_residual_vector, write_files)
        if self.write_files:
            self._q_space = df.FunctionSpace(self.fen.mesh, "DG", DG_degree)
            self.D = df.Function(self._q_space, name='Damage')
            if self.fen.hist_storage=='quadrature':
                self.K = df.Function(self._q_space, name='Kappa')

    def __call__(self, t, logger):
        super().__call__(t, logger)
        if self.write_files:
            ebar_plot = self.fen.u_mix.split()[1]
            self.xdmf.write(ebar_plot, t)
            if self.fen.hist_storage=='quadrature':
                df.project(v=self.fen.u_K_current, V=self._q_space, function=self.K) # projection
                self.xdmf.write(self.K, t)
            else:
                self.xdmf.write(self.fen.u_K_current, t) # This does not work for "u_K_current" being in quadrature space
            D_ufl = self.fen.mat.gK.g(self.fen.u_K_current)
            df.project(v=D_ufl, V=self._q_space, function=self.D) # projection
            self.xdmf.write(self.D, t)

class PostProcessPlastic(PostProcess):
    def __init__(self, fen, _name='', out_path=None, reaction_dofs=None, log_residual_vector=False, write_files=True, DG_degree=1):
        super().__init__(fen, _name, out_path, reaction_dofs, log_residual_vector, write_files)
        if self.write_files:
            elem_dg0_k = df.FiniteElement("DG", self.fen.mesh.ufl_cell(), degree=DG_degree)
            elem_dg0_ss = df.VectorElement("DG", self.fen.mesh.ufl_cell(), degree=DG_degree, dim=self.fen.mat.ss_dim)
            self.i_k = df.FunctionSpace(self.fen.mesh, elem_dg0_k)
            self.i_ss = df.FunctionSpace(self.fen.mesh, elem_dg0_ss)
            
            self.sig = df.Function(self.i_ss, name='Stress')
            self.eps_p = df.Function(self.i_ss, name='Cumulated plastic strain')
            self.K = df.Function(self.i_k, name='Cumulated Kappa')

    def __call__(self, t, logger):
        super().__call__(t, logger)
        if self.write_files:
            ### project from quadrature space to DG-0 space
            df.project(v=self.fen.sig, V=self.i_ss, function=self.sig)
            df.project(v=self.fen.eps_p, V=self.i_ss, function=self.eps_p)
            df.project(v=self.fen.kappa, V=self.i_k, function=self.K)
            ### write projected values to xdmf-files
            self.xdmf.write(self.K, t)
            self.xdmf.write(self.sig, t)
            self.xdmf.write(self.eps_p, t)

class PostProcessPlasticGDM(PostProcess):
    def __init__(self, fen, _name='', out_path=None, reaction_dofs=None, log_residual_vector=False, write_files=True, DG_degree=1):
        super().__init__(fen, _name, out_path, reaction_dofs, log_residual_vector, write_files)
        if self.write_files:
            # DG spaces
            elem_dg0_k = df.FiniteElement("DG", self.fen.mesh.ufl_cell(), degree=DG_degree)
            elem_dg0_ss = df.VectorElement("DG", self.fen.mesh.ufl_cell(), degree=DG_degree, dim=self.fen.mat.ss_dim)
            self.i_k = df.FunctionSpace(self.fen.mesh, elem_dg0_k)
            self.i_ss = df.FunctionSpace(self.fen.mesh, elem_dg0_ss)
            
            # damage-related quantities
            self.sig = df.Function(self.i_ss, name='Actual (damaged) stress')
            self.ebar_plot = self.fen.u_mix.split()[1]
            self.ebar_plot.rename('Nonlocal strain', 'Nonlocal strain')
            self.Kd = df.Function(self.i_k, name='Kappa_damage (cumulated)')
            self.D = df.Function(self.i_k, name='Damage')
            # plasticity-related quantities
            self.eps_p = df.Function(self.i_ss, name='Cumulated plastic strain')
            self.Kp = df.Function(self.i_k, name='Kappa_plastic (cumulated)')
    
    def __call__(self, t, logger):
        super().__call__(t, logger)
        if self.write_files:
            ### project from quadrature space to DG-0 space
            df.project(v=self.fen.q_sigma, V=self.i_ss, function=self.sig)
            df.project(v=self.fen.q_k, V=self.i_k, function=self.Kd)
            df.project(v=self.fen.mat.gK.g(self.fen.q_k), V=self.i_k, function=self.D)
            df.project(v=self.fen.q_eps_p, V=self.i_ss, function=self.eps_p)
            df.project(v=self.fen.q_k_plastic, V=self.i_k, function=self.Kp)
            
            ### write projected values to xdmf-files
            self.xdmf.write(self.sig, t)
            self.xdmf.write(self.ebar_plot, t)
            self.xdmf.write(self.Kd, t)
            self.xdmf.write(self.D, t)
            self.xdmf.write(self.eps_p, t)
            self.xdmf.write(self.Kp, t)

class DOFsForJacobianPropagation:
    """
    Suiting GDM problem (particularly with regards to Ebar).
    """
    def __init__(self, free=None, residual=None, imposed=None, Ebar=None):
        self.fr = free # At which F=0 is fulfilled (residual vanishes).
        self.res = residual # At which we have non-zero residuals.
        self.imp = imposed # At which we impose prescribed values.
        self.e = Ebar # for GDM problem is DOFs of nonlocal equivalent strain.
        try:
            # NOTE: res and imp DOFs are most likely the same but differently sorted. So, we check this and return a Warning, otherwise.
            _a = self.res[:] # a copy
            _b = self.imp[:] # a copy
            if _a.sort() != _b.sort():
                print('DOFsForJacobianPropagation --- WARNING --- :\n    There is NO one-to-one map between provided imposed and residuals DOFs for the propagation of Jacobian.')
        except:
            print('DOFsForJacobianPropagation --- WARNING --- :\n    At least one of residual and/or free DOFs is None.')
        try:
            if len(set(self.imp).intersection(set(self.fr))) > 0:
                raise ValueError('DOFsForJacobianPropagation --- ERROR --- :\n    There exist some common values in imposed and free DOFs.')
            if not set(self.dof.e).issubset(set(self.dof.fr)):
                raise ValueError('DOFsForJacobianPropagation --- ERROR --- :\n    The free DOFs does not contain all of Ebar DOFs.')
            if len(set(self.res).intersection(set(self.fr))) > 0:
                print('DOFsForJacobianPropagation --- WARNING --- :\n    There exist some common values in residuals and free DOFs.')
        except:
            pass
    
class PropagateJacobian:
    """
    This has been designed for a gradient damage model/problem, but can also be adjusted for similar problems with history variables (K).
    u:  solution field at total DOFs
    uu: solution field at displacement DOFs (main field)
    ue: solution field at Ebar DOFs (second field)
    D:  full derivative
    d:  partial derivative
    """
    def __init__(self, fen, dof, dF_dps=None, _Fu=False, _J_Fu=False, gs=[], _gs=False):
        self.fen = fen # FEniCS problem object
        self.hist_size = self.fen.i_K.dim()
        self.dof = dof # A DOFsForJacobianPropagation instance
        if dF_dps is None:
            dF_dps = self.fen.dF_dps_vals
        self.dF_dps = dF_dps # dictionary of evaluated derivatives of total "F" of self.fen with respect to desired parameters (type: np.array).
            ## IMPORTANT: It is assumed that self.dF_dps will always be updated according to the last solution at any time (t).
        self._Fu = _Fu # a flag for whether we compute Fu (residuals at self.dof.res) at different checkpoints
        self._J_Fu = _J_Fu # a flag deciding on whether we propagate D_Fu_d_u, where Fu is the displacement-related portion of governing equation and u is displacement. 
        
        if not fen.jac_prep:
            raise AssertionError('The FEniCS problem must have prepared quantities (tangential K and other derivatives) needed for propagation of Jacobian.')
        if dF_dps=={} and _J_Fu==False:
            raise AssertionError('Please specify with respect to which quantities Jacobian(s) should be computed. This can be done by specifying proper dF_dps or by setting _J_Fu=True.')
        if dF_dps!={}:
            if self.dof.fr is None:
                raise AssertionError('For propagation of Du_dps and Dgs_dps, all free dofs (self.dof.fr) must have been specified.')
        if _J_Fu:
            if (self.dof.fr is None) or (self.dof.res is None) or (self.dof.imp is None) or (self.dof.e is None):
                raise AssertionError('For propagation of J_Fu, degrees of freedom of free, residuals, imposed and Ebar (self.dof.fr, self.dof.res, dof.imp and self.dof.e) must have been specified.')
        
        self.checkpoints = [] # will be assigned as soon as a time-stepper incremental solution is executed.
        self.Du_dps_checked = {}
        self.DK0_dps_checked = {}
        self.Du_dps = {}
        self.last_DK0_dps = {}
        self.DK0_dps = {}
        
        self.gs = gs # list of callables each being for a particular forward-model's output, which gets "t"(time) as input argumet and returns: g, dg_dps, dg_du, dg_dK
        self._gs = _gs # a flag for whether we compute gs (forward-models' outputs) at different checkpoints
        for _g in self.gs:
            assert _g.pars_keys == self.dF_dps.keys() # The same parameters must be considered in "self.g" and "self.dF_dps"
        self.Dgs_dps = len(self.gs) * [{}] # a list of dictionaries, each dictionary being for one "g" of self.gs and containing Dg_dps based on "g.pars_keys". Each entry of dictionary is the corresponding "Dg_dp" at last time.
        self.Dgs_dps_checked = len(self.gs) * [{}] # a list of dictionaries, each dictionary being for one "g" of self.gs and containing Dg_dps based on "g.pars_keys". Each entry of dictionary is again a list of "Dg_dp" at checkpoints.
        
    def __call__(self, t, logger):
        ## "t" and "logger" are given as inputs just to be consistent with call methods of other PostProcessor classes
        if t==0:
            ## INITIATIONs
                ## Especially the items which will be appended later on, must be refreshed to empty!. The reason is that, the same object is used many times to propagate the Jacobian.
            for _k in self.dF_dps.keys():
                self.DK0_dps[_k] = np.zeros(self.hist_size) # refresh
            if self._Fu:
                block_size = len(self.checkpoints)
                self.Fu_checked = block_size * [None]
            if self._J_Fu:
                ## initiate the global BLOCK matrix of self.J_Fu_global as a list of lists
                    ## self.J_Fu_global[i][j] refers to the BLOCK matrix of J_Fi_uj
                block_size = len(self.checkpoints)
                self.J_Fu_global = np.block([block_size * [None] for i in range(block_size)])
                self.dKold_dK_checked = block_size * [None] # i-th entry is: the derivative of very-last Kappa w.r.t. Kappa at i-th checked time-step
                self.dK_checked_duu_checked = block_size * [None] # i-th entry is the derivative of current Kappa w.r.t. displacement field at i-th checked time-step
            if self._gs:
                self.gs_checked = []
                for egal in range(len(self.gs)):
                    self.gs_checked.append([])
            
            for _k in self.dF_dps.keys():
                self.Du_dps_checked[_k] = [] # empty list whose each entry will become for one checkpoint
                self.DK0_dps_checked[_k] = [] # empty list whose each entry will become for one checkpoint
            for ig, _g in enumerate(self.gs):
                for _k in _g.pars_keys:
                    self.Dgs_dps_checked[ig][_k] = []
        else:
            ## quantities needed in each case: (self.dF_dps is not {}) and (self._J_Fu=True)
            dF_du = self.fen.K_t.array()
            dF_dK0 = self.fen.dF_dK0
            dKnew_du = self.fen.dKmax_du
            dKnew_dK0 = self.fen.dKmax_dK0 # a 1-D np.array containing the diagonal terms of the actual dKnew_dK0
            
            if self._Fu:
                for i, tt in enumerate(self.checkpoints):
                    if abs(t - tt) < 1e-9:
                        self.Fu_checked[i] = df.assemble(self.fen.get_F_and_u()[0]).get_local()[self.dof.res]
                        break
            
            if self.dF_dps!={}: # propagation of Jacobian w.r.t. the parameters related to the provided "self.dF_dps"
                
                gs=[]; dgs_dps=[]; dgs_du=[]; dgs_dK=[]
                for ig, _g in enumerate(self.gs):
                    g_i, dg_dps_i, dg_du_i, dg_dK_i = _g(t)
                    gs.append(g_i)
                    dgs_dps.append(dg_dps_i)
                    dgs_du.append(dg_du_i)
                    dgs_dK.append(dg_dK_i)
                    
                    if self._gs:
                        for tt in self.checkpoints:
                            if abs(t - tt) < 1e-9:
                                self.gs_checked[ig].append(g_i)
                
                for _k, _d in self.dF_dps.items():
                    ### derivative of F w.r.t. parameter
                    dF_dp = _d[self.dof.fr]
                    
                    ### Du_dps
                    A = dF_du[np.ix_(self.dof.fr, self.dof.fr)]
                    B = dF_dp + dF_dK0[self.dof.fr] @ self.DK0_dps[_k]
                    self.Du_dps[_k] = - np.linalg.solve(A, B)
                    
                    ### Update self.DK0_dps (needed for the next time-step)
                    self.last_DK0_dps[_k] = self.DK0_dps[_k] # before updating (replacing) with the new one
                    self.DK0_dps[_k] = dKnew_du[:, self.dof.fr] @ self.Du_dps[_k] + np.multiply(dKnew_dK0, self.DK0_dps[_k])
                    
                    ## Analytical Dg_dp
                    for ig in range(len(self.gs)):
                        self.Dgs_dps[ig][_k] = compute_Dg_dp(dgs_dps[ig][_k], dgs_du[ig], dgs_dK[ig] \
                                                            , self.Du_dps[_k], self.last_DK0_dps[_k], self.dof.fr)
                    
                    for tt in self.checkpoints:
                        if abs(t - tt) < 1e-9:
                            self.Du_dps_checked[_k].append(self.Du_dps[_k])
                            self.DK0_dps_checked[_k].append(self.DK0_dps[_k])
                            for ig, _ in enumerate(self.gs):
                                self.Dgs_dps_checked[ig][_k].append(self.Dgs_dps[ig][_k])
        
            if self._J_Fu: # propagation of Jacobian of Fu w.r.t. displacement field over time-steps
                _ids = [self.dof.fr.index(q) for q in self.dof.e] # self.dof.fr[_ids] = self.dof.e
                for i, tt in enumerate(self.checkpoints):
                    if abs(t - tt) < 1e-9:
                        A = dF_du[np.ix_(self.dof.res, self.dof.imp)]
                        B = dF_du[np.ix_(self.dof.res, self.dof.fr)]
                        C = dF_du[np.ix_(self.dof.fr, self.dof.imp)]
                        D = dF_du[np.ix_(self.dof.fr, self.dof.fr)]
                        
                        ## Diagonal blocks
                        minus_invD_C = - np.linalg.solve(D, C) # This is in fact d_u_free_d_u_imposed.
                        self.J_Fu_global[i][i] = A + np.dot(B , minus_invD_C) # the sub-matrix located on the diagonal of the whole block matrix
                        
                        ## Off-diagonal blocks
                        dFu_dK0_1 = dF_dK0[self.dof.res, :] # partial
                        A2 = dF_du[np.ix_(self.dof.res, self.dof.fr)]
                        C2 = dF_dK0[self.dof.fr, :]
                        D2 = D
                        minus_invD2_C2 = - np.linalg.solve(D2, C2) # is "d_u_free_d_K0"
                        dFu_dK0_2 = A2 @ minus_invD2_C2
                        DFu_DK0 = dFu_dK0_1 + dFu_dK0_2
                        ## Before update of self.dKold_dK_checked and self.dK_checked_duu_checked (for being used for next time-steps)
                        ## , we use it for computing off-diagonal blocks
                        for j in range(i): # previous checkpoints
                            DKold_duu_checked = self.dKold_dK_checked[j] @ self.dK_checked_duu_checked[j]
                            self.J_Fu_global[i][j] = DFu_DK0 @ DKold_duu_checked
                        
                        ## Update self.dK_checked_duu_checked
                        # We use smaller sub-matrices to multiply.
                            # It is just the same as version-2 (below) since the ommited part of "dKnew_du" (derivative w.r.t. duu) is zero.
                        d_ue_d_uu = minus_invD_C[_ids, :] # This is in fact d_u_e_d_u_imposed.
                        self.dK_checked_duu_checked[i] = dKnew_du[:, self.dof.e] @ d_ue_d_uu # used for next checkpoint
                            ## Version-2 (for being documented): using full matrices.
                            # d_uf_d_uu = minus_invD_C[:, :] # This is in fact d_u_free_d_u_imposed.
                            # self.dK_checked_duu_checked[i] = dKnew_du[:, self.dof.fr] @ d_uf_d_uu # used for next checkpoint
                        break
                
                ## Update (already initiated) self.dKold_dK_checked for next time-steps
                new_in_chain = np.diag(dKnew_dK0)
                for i, d in enumerate(self.dKold_dK_checked):
                    if d is not None:
                        self.dKold_dK_checked[i] = new_in_chain @ d
                
                ## Initiate self.dKold_dK_checked
                for i, tt in enumerate(self.checkpoints):
                    if abs(t - tt) < 1e-9:
                        self.dKold_dK_checked[i] = np.eye(self.hist_size)
                            # For the very next time-increment, this is 1, since at that time-step we have K_old=K_checked.
                            # For the rest of next time-steps, it will be updated according to dKnew_dK0 (see above)
                        break

