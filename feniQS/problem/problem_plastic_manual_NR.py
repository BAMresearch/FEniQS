from feniQS.problem.problem import *
from feniQS.fenics_helpers.fenics_functions import LocalProjector
from feniQS.material.constitutive_plastic import *

pth_problem_plastic_manual_NR = CollectPaths('./feniQS/problem/problem_plastic_manual_NR.py')
pth_problem_plastic_manual_NR.add_script(pth_problem)
pth_problem_plastic_manual_NR.add_script(pth_fenics_functions)
pth_problem_plastic_manual_NR.add_script(pth_constitutive_plastic)
    
class FenicsPlastic(FenicsProblem):
    def __init__(self, mat, mesh, fen_config, dep_dim=None):
        super().__init__(mat, mesh, fen_config, dep_dim)
    
    def build_variational_functionals(self, f=None, integ_degree=None, expr_sigma_scale=1.):
        self.hist_storage = 'quadrature' # always is quadrature
        if integ_degree is None:
            integ_degree = self.shF_degree_u + 1
        f = super().build_variational_functionals(f, integ_degree) # includes a call for discretize method
        
        df.parameters["form_compiler"]["representation"] = "quadrature"
        import warnings
        from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
        warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
        
        self.a_Newton = expr_sigma_scale * df.inner(eps_vector(self.v, self.mat.constraint), df.dot(self.Ct, eps_vector(self.v_u, self.mat.constraint))) * self.dxm
        self._F_int = expr_sigma_scale * df.inner(eps_vector(self.v_u, self.mat.constraint), self.sig1) * self.dxm
    
    def get_solution_field(self):
        return self.u_u

    def discretize(self):
        ### Nodal spaces / functions
        if self.dep_dim == 1:
            elem_u = df.FiniteElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u)
        else:
            elem_u = df.VectorElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u, dim=self.dep_dim)
        self.i_u = df.FunctionSpace(self.mesh, elem_u)
        # Define functions
        self.u_u = df.Function(self.i_u, name="Converged displacement at the end of time-step")
        self.u1 = df.Function(self.i_u, name="Current displacement at last global NR iteration")
        self.du = df.Function(self.i_u, name="Correction in global NR iteration")
        self.v = df.TrialFunction(self.i_u)
        self.v_u = df.TestFunction(self.i_u)
        
        ### Quadrature spaces / functions
        # for sigma and strain
        elem_ss = df.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, dim=self.mat.ss_dim, quad_scheme="default")
        self.i_ss = df.FunctionSpace(self.mesh, elem_ss)
        # for scalar history variables on gauss points (for now: scalar)
        elem_scalar_gauss = df.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, quad_scheme="default")
        self.i_hist = df.FunctionSpace(self.mesh, elem_scalar_gauss)
        # for tangent matrix
        elem_tensor = df.TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, shape=(self.mat.ss_dim, self.mat.ss_dim), quad_scheme="default")
        i_tensor = df.FunctionSpace(self.mesh, elem_tensor)
        
        self.ngauss = self.i_hist.dim() # get total number of gauss points
        
        # Define functions based on Quadrature spaces
        self.sig1 = df.Function(self.i_ss, name="Current Stress after last global NR iteration")
        self.eps1 = df.Function(self.i_ss, name="Current strain at last global NR iteration")
        self.sig = df.Function(self.i_ss, name="Converged stress at the end of time-step")
        self.eps = df.Function(self.i_ss, name="Converged strain at the end of time-step")
        
        self.kappa1 = np.zeros(self.ngauss) # Current history variable at last global NR iteration
        self.kappa = df.Function(self.i_hist, name="Converged cumulated history variable at the end of time-step")
        
        self.d_eps_p_num = np.zeros((self.ngauss, self.mat.ss_dim)) # Change of plastic strain at last global NR iteration
        self.eps_p = df.Function(self.i_ss, name="Converged cumulated plastic strain at the end of time-step")
        
        # Define and initiate tangent operator (elasto-plastic stiffness matrix)
        self.Ct = df.Function(i_tensor, name="Consistent tangent operator")
        self.Ct_num = np.tile(self.mat.D.flatten(), self.ngauss).reshape((self.ngauss, self.mat.ss_dim**2)) # initial value is the elastic stiffness at all Gauss-points
        self.Ct.vector().set_local(self.Ct_num.flatten()) # assign the helper "Ct_num" to "Ct"
        self.sig1_num = np.zeros((self.ngauss, self.mat.ss_dim)) # this is just helpfull for assigning to self.sig1
    
    def build_solver(self, solver_options=None, time_varying_loadings=[], tol=1e-12):
        FenicsProblem.build_solver(self, time_varying_loadings)
        self.solver = 'Manually implemented' # We perform a NR-solver manually in self.solve().
        self.projector_eps = LocalProjector(eps_vector(self.u1, self.mat.constraint), self.i_ss, self.dxm)
        if len(self.bcs_DR + self.bcs_DR_inhom) == 0:
            print('WARNING: No boundary conditions have been set to the FEniCS problem.')
        self.assembler = df.SystemAssembler(self.a_Newton, self.get_F_and_u()[0], self.bcs_DR + self.bcs_DR_inhom)
        self.A = df.PETScMatrix() # to initiate
        self.b = df.PETScVector() # to initiate
        self.sol_tol = tol
    
    def solve(self, t=0, max_iters=50, allow_nonconvergence_error=True):
        FenicsProblem.solve(self, t)
        conv = False
        it = 0
        self.u1.assign(self.u_u)
        self.assembler.assemble(self.A)
        self.assembler.assemble(self.b, self.u1.vector())
        nRes0 = self.b.norm("l2")
        nRes = nRes0
        print("Time increment = ", str(t))
        print("    Residual_" + str(it) + " =", nRes)
        if nRes0==0: # trivial solution
            return (0, True)
        else:
            while nRes / nRes0 > self.sol_tol and it < max_iters:
                # solve for current increment in the iteration
                df.solve(self.A, self.du.vector(), self.b)
                # update the current displacement
                self.u1.assign(self.u1 - self.du)
                # project the corresponding mandel strain to quadrature space
                self.projector_eps(self.eps1)
                # compute correct stress and Ct based on the updated strain "self.eps1" (perform return-mapping, if needed)
                self._correct_stress_and_Ct()
                # update Newton system (residual and A matrix) for next iteration
                self.assembler.assemble(self.A)
                self.assembler.assemble(self.b, self.u1.vector())
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
    
    def _correct_stress_and_Ct(self):
        """
        given:
            self.eps1
        , we perform:
            the update of stress and Ct
        """
        # compute the incremental strain
        d_eps = self.eps1.vector().get_local().reshape((-1, self.mat.ss_dim)) \
            - self.eps.vector().get_local().reshape((-1, self.mat.ss_dim))
        
        # compute trial (predicted) stress
        sig_tr = self.sig.vector().get_local().reshape((-1, self.mat.ss_dim)) \
            + d_eps @ self.mat.D
            # ---- ??? better to be replaced with Ct if loading from an already yielded point (must be performed point-wise)
        
        # perform return-mapping (if needed) per individual Gauss-points
        for i in range(self.ngauss):
            sig_cr, Ct, k, d_eps_p = self.mat.correct_stress(sig_tr=sig_tr[i], k0=self.kappa.vector()[i])
            # assignments:
            self.kappa1[i] = k # update history variable(s)
            self.sig1_num[i] = sig_cr
            self.Ct_num[i] = Ct.flatten()
            self.d_eps_p_num[i] = d_eps_p # store change in the cumulated plastic strain
        
        # assign the helper "sig1_num" to "sig1"
        self.sig1.vector().set_local(self.sig1_num.flatten())
        # assign the helper "Ct_num" to "Ct"
        self.Ct.vector().set_local(self.Ct_num.flatten())
    
    def _todo_after_convergence(self):
        self.u_u.assign(self.u1)
        self.kappa.vector()[:] = self.kappa1
        self.sig.vector()[:] = self.sig1.vector().get_local()
        self.eps.vector()[:] = self.eps1.vector().get_local()
        self.eps_p.vector()[:] = self.eps_p.vector()[:] + self.d_eps_p_num.flatten()
        # postprocessing results
    
    def get_i_full(self):
        return self.i_u
    
    def get_uu(self):
        return self.u_u
    
    def get_iu(self, _collapse=True):
        return self.i_u
    
    def reset_fields(self, u0=0.0):
        # u0 can be a vector of the same length as self.u_u
        self.u_u.vector()[:] = u0
    