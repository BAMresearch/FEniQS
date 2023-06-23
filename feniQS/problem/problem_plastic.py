from feniQS.problem.problem import *
from feniQS.fenics_helpers.fenics_functions import LocalProjector
from feniQS.material.constitutive_plastic import *

pth_problem_plastic = CollectPaths('./feniQS/problem/problem_plastic.py')
pth_problem_plastic.add_script(pth_problem)
pth_problem_plastic.add_script(pth_fenics_functions)
pth_problem_plastic.add_script(pth_constitutive_plastic)
    
class FenicsPlastic(FenicsProblem, df.NonlinearProblem):
    def __init__(self, mat, mesh, fen_config, dep_dim=None):
        FenicsProblem.__init__(self, mat, mesh, fen_config, dep_dim)
        df.NonlinearProblem.__init__(self)
    
    def build_variational_functionals(self, f=None, integ_degree=None):
        self.hist_storage = 'quadrature' # always is quadrature
        if integ_degree is None:
            integ_degree = self.shF_degree_u + 1
        f = FenicsProblem.build_variational_functionals(self, f, integ_degree) # includes a call for discretize method
        
        df.parameters["form_compiler"]["representation"] = "quadrature"
        import warnings
        from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
        warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
        
        self.a_Newton = df.inner(eps_vector(self.v, self.mat.constraint), df.dot(self.Ct, eps_vector(self.u_, self.mat.constraint))) * self.dxm
        self.res = ( df.inner(eps_vector(self.u_, self.mat.constraint), self.sig) - df.inner(f, self.u_) ) * self.dxm
        
    def discretize(self):
        ### Nodal spaces / functions
        if self.dep_dim == 1:
            elem_u = df.FiniteElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u)
        else:
            elem_u = df.VectorElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u, dim=self.dep_dim)
        self.i_u = df.FunctionSpace(self.mesh, elem_u)
        # Define functions
        self.u = df.Function(self.i_u, name="Displacement at last global NR iteration")
        self.v = df.TrialFunction(self.i_u)
        self.u_ = df.TestFunction(self.i_u)
        
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
        self.sig = df.Function(self.i_ss, name="Stress")
        self.eps = df.Function(self.i_ss, name="Strain")
        self.sig_num = np.zeros((self.ngauss, self.mat.ss_dim)) # this is just helpfull for assigning to self.sig
        
        self.kappa1 = np.zeros(self.ngauss) # Current history variable at last global NR iteration
        self.kappa = df.Function(self.i_hist, name="Cumulated history variable")
        
        self.d_eps_p_num = np.zeros((self.ngauss, self.mat.ss_dim)) # Change of plastic strain at last global NR iteration
        self.eps_p = df.Function(self.i_ss, name="Cumulated plastic strain")
        
        # Define and initiate tangent operator (elasto-plastic stiffness matrix)
        self.Ct = df.Function(i_tensor, name="Tangent operator")
        self.Ct_num = np.tile(self.mat.D.flatten(), self.ngauss).reshape((self.ngauss, self.mat.ss_dim**2)) # initial value is the elastic stiffness at all Gauss-points
        self.Ct.vector().set_local(self.Ct_num.flatten()) # assign the helper "Ct_num" to "Ct"
    
    def build_solver(self, time_varying_loadings=[], tol=1e-12, solver=None):
        FenicsProblem.build_solver(self, time_varying_loadings)
        self.projector_eps = LocalProjector(eps_vector(self.u, self.mat.constraint), self.i_ss, self.dxm)
        if len(self.bcs_DR + self.bcs_DR_inhom) == 0:
            print('WARNING: No boundary conditions have been set to the FEniCS problem.')
        self.assembler = df.SystemAssembler(self.a_Newton, self.res, self.bcs_DR + self.bcs_DR_inhom)
        if solver is None:
            solver = df.NewtonSolver()
            solver.parameters['maximum_iterations'] = 14
        self.solver = solver
    
    def F(self, b, x):
        # project the corresponding strain to quadrature space
        self.projector_eps(self.eps)
        # compute correct stress and Ct based on the updated strain "self.eps" (perform return-mapping, if needed)
        self._correct_stress_and_Ct()
        # update the solver's residual
        self.assembler.assemble(b, x)

    def J(self, A, x):
        # update the solver's tangent operator
        self.assembler.assemble(A)

    def solve(self, t=0.0, max_iters=20, allow_nonconvergence_error=True):
        FenicsProblem.solve(self, t)
        _it, conv = self.solver.solve(self, self.u.vector())
        if conv:
            print(f"    The time step t={t} converged after {_it} iteration(s).")
            self._todo_after_convergence()
        else:
            print(f"    The time step t={t} did not converge.")
        return (_it, conv)
    
    def _correct_stress_and_Ct(self):
        """
        given:
            self.eps
        , we perform:
            the update of stress and Ct
        """
        eps = self.eps.vector().get_local().reshape((-1, self.mat.ss_dim))
        eps_p = self.eps_p.vector().get_local().reshape((-1, self.mat.ss_dim))
        
        # perform return-mapping (if needed) per individual Gauss-points
        for i in range(self.ngauss):
            sig_tr_i = np.atleast_1d(self.mat.D @ (eps[i] - eps_p[i]))
            sig_cr, Ct, k, d_eps_p = self.mat.correct_stress(sig_tr=sig_tr_i, k0=self.kappa.vector()[i])
            # assignments:
            self.kappa1[i] = k # update history variable(s)
            self.sig_num[i] = sig_cr
            self.Ct_num[i] = Ct.flatten()
            self.d_eps_p_num[i] = d_eps_p # store change in the cumulated plastic strain
        
        # assign the helper "sig_num" to "sig"
        self.sig.vector().set_local(self.sig_num.flatten())
        # assign the helper "Ct_num" to "Ct"
        self.Ct.vector().set_local(self.Ct_num.flatten())
    
    def _todo_after_convergence(self):
        self.kappa.vector()[:] = self.kappa1
        self.eps_p.vector()[:] = self.eps_p.vector()[:] + self.d_eps_p_num.flatten()
        
    def get_F_and_u(self):
        return self.res, self.u
    
    def get_uu(self):
        return self.u
    
    def get_i_full(self):
        return self.i_u
    