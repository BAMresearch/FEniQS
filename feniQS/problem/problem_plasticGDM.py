"""
This implements the section (2.3) of the paper:
    https://www.sciencedirect.com/science/article/pii/S099775389900114X
, which models gradient damage model subjected to plasticity hardening.
"""

import sys
if './' not in sys.path:
    sys.path.append('./')

from feniQS.problem.problem import *
from feniQS.problem.problem_plastic import *

pth_problem_plasticGDM = CollectPaths('problem_plasticGDM.py')
pth_problem_plasticGDM.add_script(pth_problem)
pth_problem_plasticGDM.add_script(pth_problem_plastic)

def damage_degrade(sig_eff, D_eff, gK, K, dK_debar):
    """
    sig_eff:
        effective (undamaged) stress vector
    D_eff:
        effective (undamaged) (elastic) tangent operator
    gK:
        damage law object with necessary callables: g_eval, dg_dK_eval
    K:
        damage-related history variable kappa
    dK_debar:
        derivative of Kappa w.r.t. nonlocal strain (Ebar)
    
    Returns:
        actual (damaged) sigma
        derivative of actual sigma w.r.t. total_strain : actual (damaged) tangent operator
        derivative of actual sigma w.r.t. nonlocal strain (Ebar)
    """
    g = gK.g_eval(K)
    dg_dK = gK.dg_dK_eval(K)
    return (1 - g) * sig_eff, (1 - g) * D_eff, - sig_eff * dg_dK * dK_debar

class FenicsPlasticGDM(FenicsProblem):
    def __init__(self, mat_gdm, mat_plastic, mesh, fen_config, dep_dim=None, K_current=df.Constant(0.0)):
        super().__init__(mat_gdm, mesh, fen_config, dep_dim) # self.mat is related to GDM
        
        # concerning GDM
        self.shF_degree_ebar = fen_config.shF_degree_ebar
        # self.K_current = K_current
        
        # concerning plasticity
        self.mat_plastic = mat_plastic
    
    def build_variational_functionals(self, f=None, integ_degree=None, expr_sigma_scale=1):
        self.hist_storage = 'quadrature' # always is quadrature
        if integ_degree is None:
            integ_degree = max(self.shF_degree_u, self.shF_degree_ebar) + 1
        f = super().build_variational_functionals(f, integ_degree) # includes a call for discretize method
        
        df.parameters["form_compiler"]["representation"] = "quadrature"
        import warnings
        from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
        warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
        
        eps_u = eps_vector(self.u_u, self.mat.constraint) # regarding function
        eps_v = eps_vector(self.v_u, self.mat.constraint) # regarding test function
        eps_d = eps_vector(self.d_u, self.mat.constraint) # regarding trial function
        
        ### nonlinear terms in self.R (needed to be updated/resolved):
            # self.q_sigma
            # self.q_eeq
        self.R = expr_sigma_scale * df.inner(eps_v, self.q_sigma) * self.dxm
        self.R += self.v_ebar * (self.u_ebar - self.q_eeq) * self.dxm
        self.R += df.dot(df.grad(self.v_ebar), self.mat.c * df.grad(self.u_ebar)) * self.dxm
        
        ### nonlinear terms in self.dR (needed to be updated/resolved):
            # self.q_dsigma_deps
            # self.q_dsigma_de
            # self.q_deeq_deps
        self.dR = expr_sigma_scale * df.inner(eps_d, self.q_dsigma_deps * eps_v) * self.dxm
        self.dR += expr_sigma_scale * self.d_ebar * df.dot(self.q_dsigma_de, eps_v) * self.dxm
        self.dR += df.inner(eps_d, - self.q_deeq_deps * self.v_ebar) * self.dxm
        self.dR += (self.d_ebar * self.v_ebar + df.dot(df.grad(self.d_ebar), self.mat.c * df.grad(self.v_ebar)) ) * self.dxm
        
        self._resolve_damage() # VERY IMPORTANT
        
        ### Set projectors ########### better to be in build_solver
        self.projector_eps = LocalProjector(eps_u, self.i_ss, self.dxm)
        self.projector_ebar = LocalProjector(self.u_ebar, self.i_hist, self.dxm)
        
    def discretize(self):
        ### Nodal spaces / functions
        if self.dep_dim == 1:
            elem_u = df.FiniteElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u)
        else:
            elem_u = df.VectorElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u, dim=self.dep_dim)
        ## OR generally: elem_u = df.VectorElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u, dim=self.dep_dim)
        
        elem_ebar = df.FiniteElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_ebar)
        elem_mix = df.MixedElement([elem_u, elem_ebar])
        self.i_mix = df.FunctionSpace(self.mesh, elem_mix)
        self.i_u, self.i_ebar = self.i_mix.split()
        # self.u_mix = df.Function(self.i_mix, name="Mixed fields")
        # self.u_u, self.u_ebar = df.split(self.u_mix)
        self.u_mix = df.Function(self.i_mix, name='Current mixed fields at last global NR iteration')
        self.u_u, self.u_ebar = df.split(self.u_mix)
        self.du_mix = df.Function(self.i_mix, name="Correction in global NR iteration")
        
        v_mix  = df.TestFunction(self.i_mix)
        self.v_u, self.v_ebar = df.split(v_mix)
        d_mix  = df.TrialFunction(self.i_mix)
        self.d_u, self.d_ebar = df.split(d_mix)
            ### OR:
            # self.v_u, self.v_ebar = df.TestFunctions(self.i_mix)
            # self.d_u, self.d_ebar = df.TrialFunctions(self.i_mix)
        
        ### Quadrature spaces
        # for sigma and strain
        elem_ss = df.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, dim=self.mat.ss_dim, quad_scheme="default")
        self.i_ss = df.FunctionSpace(self.mesh, elem_ss)
        # for scalar history variables on gauss points (for now: scalar)
        elem_scalar_gauss = df.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, quad_scheme="default")
        self.i_hist = df.FunctionSpace(self.mesh, elem_scalar_gauss)
        # for tangent matrix
        elem_tensor = df.TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, shape=(self.mat.ss_dim, self.mat.ss_dim), quad_scheme="default")
        self.i_tensor = df.FunctionSpace(self.mesh, elem_tensor)
        
        self.ngauss = self.i_hist.dim() # get total number of gauss points
        
        ### Quadrature functions
        self.q_sigma = df.Function(self.i_ss, name="current actual (damaged) stresses")
        self.q_eps = df.Function(self.i_ss, name="current strains")
        self.q_e = df.Function(self.i_hist, name="current nonlocal equivalent strains")
        self.q_k1 = df.Function(self.i_hist, name="current history variable kappa")
        self.q_k = df.Function(self.i_hist, name="converged history variable kappa")
        self.q_eeq = df.Function(self.i_hist, name="current (local) equivalent strain (norm)")

        self.q_dsigma_deps = df.Function(self.i_tensor, name="stress-strain tangent")
        self.q_dsigma_de = df.Function(self.i_ss, name="stress-nonlocal-strain tangent")
        self.q_deeq_deps = df.Function(self.i_ss, name="equivalent-strain-strain tangent")
        
        ###### plasticity
        self.num_k1_plastic = np.zeros(self.ngauss) # Current plasticity history variable at last global NR iteration
        self.q_k_plastic = df.Function(self.i_hist, name="Cumulated plastic history variable at the end of time-step")
        self.d_eps_p_num = np.zeros((self.ngauss, self.mat.ss_dim)) # Change of plastic strain at last global NR iteration
        self.q_eps_p = df.Function(self.i_ss, name="Cumulated plastic strain at the end of time-step")
        
        # helper vectors/arrays for updating self.q_sigma, self.q_dsigma_deps, self.q_dsigma_de, self.q_deeq_deps
        self.sigma_num = np.zeros((self.ngauss, self.mat.ss_dim))
        self.dsigma_deps_num = np.zeros((self.ngauss, self.mat.ss_dim, self.mat.ss_dim))
        self.dsigma_debar_num = np.zeros((self.ngauss, self.mat.ss_dim))
        self.deeq_num = np.zeros((self.ngauss, self.mat.ss_dim))
    
    def build_solver(self, time_varying_loadings=[], tol=1e-12):
        FenicsProblem.build_solver(self, time_varying_loadings)
        self.solver = 'Manually implemented' # We perform a NR-solver manually in self.solve().
        if len(self.bcs_DR + self.bcs_DR_inhom) == 0:
            print('WARNING: No boundary conditions have been set to the FEniCS problem.')
        self.assembler = df.SystemAssembler(self.dR, self.R, self.bcs_DR + self.bcs_DR_inhom)
        self.A = df.PETScMatrix() # to initiate
        self.b = df.PETScVector() # to initiate
        self.sol_tol = tol
    
    def solve(self, t=0, max_iters=50, allow_nonconvergence_error=True):
        FenicsProblem.solve(self, t)
        conv = False
        it = 0
        self.assembler.assemble(self.A)
        self.assembler.assemble(self.b, self.u_mix.vector())
        nRes0 = self.b.norm("l2")
        nRes = nRes0
        print("Time increment = ", str(t))
        print("    Residual_" + str(it) + " =", nRes)
        if nRes0==0: # trivial solution
            return (0, True)
        else:
            while nRes / nRes0 > self.sol_tol and it < max_iters:
                # solve for current increment in the iteration
                df.solve(self.A, self.du_mix.vector(), self.b)
                # update the current displacement
                self.u_mix.assign(self.u_mix - self.du_mix)
                # project the corresponding quantities to quadrature space
                self.projector_eps(self.q_eps)
                self.projector_ebar(self.q_e)
                # resolve damage law and update relevant quantities appearing in self.R and self.dR
                self._resolve_damage()
                # update Newton system (residual and A matrix) for next iteration
                self.assembler.assemble(self.A)
                self.assembler.assemble(self.b, self.u_mix.vector())
                nRes = self.b.norm("l2")
                it += 1
                print("    Residual_" + str(it) + " =", nRes)
                if nRes / nRes0 <= self.sol_tol: # success!
                    conv = True
            if conv:
                self._todo_after_convergence()
            else:
                print("    The time step did not converge.")
            return (it, conv)
    
    def _resolve_damage(self):
        """
        This resolves the damage and updates relevant varying quantities appearing in self.R and self.dR .
        """
        eps = self.q_eps.vector().get_local().reshape((-1, self.mat.ss_dim))
        eps_p = self.q_eps_p.vector().get_local().reshape((-1, self.mat.ss_dim))
        ebar = self.q_e.vector().get_local()
        k = self.q_k.vector().get_local() # last converged kappa
        
        # vectors to be updated directly
        k1 = self.q_k1.vector()
        eeq = self.q_eeq.vector()
        
        for i in range(self.ngauss):
            k1[i] = np.maximum(k[i], ebar[i])
            dk1_debar = (ebar[i] >= k1[i]).astype(int)
            
            ### EFFECTIVE STRESS AND TANGENT MATRIX
            ### subjected to plasticity behind gradient damage
            sig_tr = np.atleast_1d(self.mat.D @ (eps[i] - eps_p[i]))
            sig_eff, D_eff, k_p, d_eps_p = self.mat_plastic.correct_stress(sig_tr=sig_tr, k0=self.q_k_plastic.vector()[i])
            self.num_k1_plastic[i] = k_p # update plastic history variable(s)
            self.d_eps_p_num[i] = d_eps_p # store change in the cumulated plastic strain
            
            ### For the sake of documentation: ###
                ## case of pure gradient damage
                # sig_eff = self.mat.D @ eps[i] # corrected stress when we have plasticity happening before damage.
                # D_eff = self.mat.D # elastoplastic stiffness matrix when we have plasticity happening before damage.
            
            self.sigma_num[i], self.dsigma_deps_num[i], self.dsigma_debar_num[i] = damage_degrade(sig_eff, D_eff, self.mat.gK, k1[i], dk1_debar)
            
            ### compute equivalent strain depending on total strain (coupling plasticity-GDM)
            eeq[i], self.deeq_num[i] = self.mat.epsilon_eq_num(eps[i])
        
        ## set helper numpy arrays to FEniCS vectors
        self.q_sigma.vector().set_local(self.sigma_num.flatten())
        self.q_dsigma_deps.vector().set_local(self.dsigma_deps_num.flatten())
        self.q_dsigma_de.vector().set_local(self.dsigma_debar_num.flatten())
        self.q_deeq_deps.vector().set_local(self.deeq_num.flatten())
    
    def _todo_after_convergence(self):
        # self._resolve_damage()
        self.q_k.assign(self.q_k1)
        self.q_k_plastic.vector()[:] = self.num_k1_plastic
        self.q_eps_p.vector()[:] = self.q_eps_p.vector()[:] + self.d_eps_p_num.flatten()
        # postprocessing results
        
    def get_F_and_u(self):
        return self.R, self.u_mix
    
    def get_uu(self, _deepcopy=True):
        return self.u_mix.split(deepcopy=_deepcopy)[0]
    
    def get_iu(self, _collapse=False):
        iu = self.i_mix.sub(0)
        if _collapse:
            iu = iu.collapse()
        return iu
    
    def get_i_full(self):
        return self.i_mix
