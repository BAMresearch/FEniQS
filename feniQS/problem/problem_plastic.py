from feniQS.problem.problem import *
from feniQS.fenics_helpers.fenics_functions import LocalProjector
from feniQS.material.constitutive_plastic import *

pth_problem_plastic = CollectPaths('./feniQS/problem/problem_plastic.py')
pth_problem_plastic.add_script(pth_problem)
pth_problem_plastic.add_script(pth_fenics_functions)
pth_problem_plastic.add_script(pth_constitutive_plastic)

class IsotropicHardeningPars(ParsBase):
    _type = 'isotropic'
    _hypotheses = ['unit', 'plastic-work']
    _laws = ['linear', 'exponential']
    def __init__(self, hypothesis='unit', law='linear', modulus=0., sig_u=None, **kwargs):
        """
        hypothesis:
            specifies how the history variable (kappa) evolves (based on other quantities such as strain)
        law:
            specifies how the yield strength (sy) of material evolves based on the initial strength (sy0)
                and the history variable kappa.
            law = 'linear':
                sy = sy0 + modulus * kappa
                This linear evolution goes for ever if 'sig_u' is None. Otherwise, as soon as 'sy' reaches 'sig_u',
                it stays at that value (constant 'sig_u'). This would mean a perfect plasticity with a yield strength
                equal to 'sig_u', which may cuase convergence issues. Thus, after this threshold of 'sig_u' is met,
                normally a much smaller modulus (e.g. 0.01 * modulus) is still taken into account.
            law = 'exponential':
                sy = sy0 + (sig_u - sy0) * (1 - exp(-modulus*kappa))
        """
        if not (hypothesis.lower() in IsotropicHardeningPars._hypotheses):
            raise ValueError(f"Possible hardening hypotheses are: {IsotropicHardeningPars._hypotheses}.")
        if not (law.lower() in IsotropicHardeningPars._laws):
            raise ValueError(f"Possible hardening laws are: {IsotropicHardeningPars._laws}.")
        self.hypothesis = hypothesis.lower()
        self.law = law.lower()
        if modulus==0.:
            print(f"\nWARNING (isotropic hardening):\n\tA zero hardening modulus deactivates any hardening.")
        self.modulus = modulus
        if self.law=='exponential' and (sig_u is None):
            raise ValueError(f"An exponential hardening law requires ultimate strength 'sig_u'.")
        self.sig_u = sig_u
        if len(kwargs)==0: # Default values are set
            pass
        else: # Get from a dictionary
            ParsBase.__init__(self, **kwargs)

class PlasticityPars(ParsBase):
    def __init__(self, pars0=None, **kwargs):
        ParsBase.__init__(self, pars0)
        if len(kwargs)==0: # Default values are set
            self.constraint = 'UNIAXIAL'
            # self.constraint = 'PLANE_STRAIN'
            # self.constraint = 'PLANE_STRESS'
            # self.constraint     = '3D'
            
            self.E_min = 0. # will be added to self.E
            self.E = 1e3
            self.nu = 0.2
            
            self.mat_type = 'plasticity' # always
            
            self.yield_surf = {'type': 'von-mises'}
            self.yield_surf['pars'] = {'sig0': 10.0}
            self.hardening_isotropic = \
                IsotropicHardeningPars(hypothesis='unit' # or 'plastic-work'
                                       , law='linear'
                                       , modulus=0.
                                       , sig_u=None).__dict__
            self.el_family = 'CG'
            self.shF_degree_u = 1
            self.integ_degree = 2
            
            self.softenned_pars = ()
                # Parameters to be converted from/to FEniCS constant (to be modified more easily)
            
            self._write_files = True
            
            self.f = None # No body force
            
            self.analytical_jac = False # whether 'preparation' is done for analytical computation of Jacobian
            
        else: # Get from a dictionary
            ParsBase.__init__(self, **kwargs)
        if self.hardening_isotropic['sig_u'] is not None:
            if self.hardening_isotropic['sig_u']<=self.yield_surf['pars']['sig0']:
                raise ValueError(f"Ultimate strength must be greater than yield strength.")

class FenicsPlastic(FenicsProblem, df.NonlinearProblem):
    def __init__(self, mat, mesh, fen_config, dep_dim=None):
        FenicsProblem.__init__(self, mat, mesh, fen_config, dep_dim)
        df.NonlinearProblem.__init__(self)
    
    def build_variational_functionals(self, f=None, integ_degree=None, expr_sigma_scale=1.):
        self.hist_storage = 'quadrature' # always is quadrature
        if integ_degree is None:
            integ_degree = self.shF_degree_u + 1
        FenicsProblem.build_variational_functionals(self, f, integ_degree) # includes discritization & building external forces
        
        df.parameters["form_compiler"]["representation"] = "quadrature"
        import warnings
        from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
        warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
        
        self.a_Newton = expr_sigma_scale * df.inner(eps_vector(self.v, self.mat.constraint), df.dot(self.Ct, eps_vector(self.v_u, self.mat.constraint))) * self.dxm
        self._F_int = expr_sigma_scale * df.inner(eps_vector(self.v_u, self.mat.constraint), self.sig) * self.dxm
    
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
        self.u_u = df.Function(self.i_u, name="Displacement at last global NR iteration")
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
    
    def build_solver(self, solver_options=None, time_varying_loadings=[]):
        FenicsProblem.build_solver(self, time_varying_loadings)
        self.projector_eps = LocalProjector(eps_vector(self.u_u, self.mat.constraint), self.i_ss, self.dxm)
        self.assembler = df.SystemAssembler(self.a_Newton, self.get_F_and_u()[0], self.bcs_DR + self.bcs_DR_inhom)
        if solver_options is None:
            solver_options = get_fenicsSolverOptions()
        self.solver = get_nonlinear_solver(solver_options=solver_options, mpi_comm=self.mesh.mpi_comm())
    
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
        _ , u = self.get_F_and_u()
        _it, conv = self.solver.solve(self, u.vector())
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
    
    def get_i_full(self):
        return self.i_u
    
    def get_uu(self):
        return self.u_u
    
    def get_iu(self, _collapse=True):
        return self.i_u
    
    def reset_fields(self, u0=0.0):
        # u0 can be a vector of the same length as self.u_u
        self.u_u.vector()[:] = u0