from feniQS.general.parameters import *
from feniQS.material.constitutive import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.fenics_helpers.fenics_expressions import *
from feniQS.problem.fenics_solvers import *

pth_problem = CollectPaths('./feniQS/problem/problem.py')
pth_problem.add_script(pth_parameters)
pth_problem.add_script(pth_constitutive)
pth_problem.add_script(pth_fenics_functions)
pth_problem.add_script(pth_fenics_expressions)
pth_problem.add_script(pth_fenics_solvers)

class ElasticPars(ParsBase):
    def __init__(self, pars0=None, **kwargs):
        ParsBase.__init__(self, pars0) # pars0 :: ParsBoxCompressed(ParsBase)
        if len(kwargs)==0: # Default values are set
            self.constraint = 'UNIAXIAL'
            # self.constraint = 'PLANE_STRAIN'
            # self.constraint = 'PLANE_STRESS'
            # self.constraint     = '3D'
            
            self.E_min          = 0. # will be added to self.E
            self.E              = 1e3
            self.nu             = 0.2
            
            self.mat_type       = 'elastic'
            
            self.el_family      = 'CG'
            self.shF_degree_u   = 1
            self.integ_degree   = 2
            
            self.softenned_pars = ['E']
                # Parameters to be converted from/to FEniCS constant (to be modified more easily)
            
            self._write_files = True
            
            self.f = None # No body force
            
            self.analytical_jac = False # whether 'preparation' is done for analytical computation of Jacobian
            
        else: # Get from a dictionary
            ParsBase.__init__(self, **kwargs)

class GDMPars(ParsBase):
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
            
            self.mat_type = 'gdm' # always
            self.damage_law = 'exp'
            # self.damage_law = 'perfect'
            # self.damage_law = 'linear'
            
            self.e0_min = 0. # will be added to self.e0
            self.e0 = 1e-4
            
            self.ef_min = 0.
            self.ef = 30e-4
            
            self.alpha = 0.99
            
            self.c_min = 10. # minimum value for c (must not be zero, since it decides if the nonlocal damage lengths is large enough compared to a given mesh)
            self.c = 0.
            
            self.el_family = 'Lagrange'
            self.shF_degree_u = 1
            self.shF_degree_ebar = 1
            self.integ_degree = 2
            
            self.softenned_pars = ('E', 'e0', 'ef', 'c')
                # Parameters to be converted from/to FEniCS constant (to be modified more easily)
            
            self._write_files = True
            
            self.f = None # No body force
            
            self.analytical_jac = False # whether 'preparation' is done for analytical computation of Jacobian
            
        else: # Get from a dictionary
            ParsBase.__init__(self, **kwargs)

class FenicsProblem():
    def __init__(self, mat, mesh, fen_config, dep_dim=None \
                 , penalty_dofs=[], penalty_weight=df.Constant(0.)):
        self.mat = mat
        self.mesh = mesh
        self.n_elements = self.mesh.num_cells()
        self.el_family = fen_config.el_family
        self.shF_degree_u = fen_config.shF_degree_u
        
        self.geo_dim = self.mesh.geometry().dim() # Geometrical dimension of mesh
        if dep_dim is None:
            self.dep_dim = self.geo_dim # The dimension of the disp. field as the dependent variable
        else:
            self.dep_dim = dep_dim
        
        assert(self.dep_dim==self.mat.dim)
        
        ## For the penalty DOFs
        self.penalty_dofs = penalty_dofs
        self.penalty_weight = penalty_weight
        
        self.bcs_DR = [] # list of Dirichlet BCs
        self.bcs_DR_measures = [] # list of measures
        self.bcs_DR_dofs = [] # list of DOFs of Dirichlet BCs
        
        self.bcs_DR_inhom = [] # list of inhomogeneous Dirichlet BCs
        self.bcs_DR_inhom_measures = [] # list of measures
        self.bcs_DR_inhom_dofs = [] # list of DOFs of inhomogeneous Dirichlet BCs
        
        self.bcs_NM_tractions = [] # list of Neumann traction functions
        self.bcs_NM_measures = [] # list of measures (the variable of integral, e.g. ds(0), ds(1), ... whcih also takes into account any particular domain) corresponding to Neumann traction functions
        self.bcs_NM_dofs = [] # list of DOFs of all Neumann BCs
        
        self.concentrated_forces = {} # A dictionary with keys: 'x', 'y' and 'z' as regards to the direction
                                      # Each value (per direction) is a list of dictionary each having the keys: 'location' and 'value'
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value) # standard self.name=value
        if name=='penalty_dofs':
            if (len(self.penalty_dofs) > 0):
                _msg = f"\n\nWARNING: A non-empty list of penalty_DOFs is set for the FEniCS problem '{self.__class__.__name__}'. Do consider:"
                _msg += f"\n\t- Updating the 'u_data' attribute as the data which drives the solution field at penalty_DOFs."
                _msg += f"\n\t- Setting proper penalty_weight.\n"
                print(_msg)
                if (not hasattr(self, 'u_data')):
                    self.u_data = df.Function(self.get_i_full(), name='U_data') # Data is generally living in full (mixed) space.
    
    @property
    def _K_t_penalty(self):
        if not hasattr(self, '__K_t_penalty'):
            self.__K_t_penalty = df.PETScVector(self.mesh.mpi_comm(), self.get_i_full().dim())
        self.__K_t_penalty[:] = 0.
        self.__K_t_penalty[self.penalty_dofs] = 2. * self.penalty_weight.values()[0]
        return self.__K_t_penalty
    
    @property
    def penalty_forces(self):
        """
        Returns vector of penalty forces only at penalty_dofs.
        """
        du = (self.get_F_and_u()[1].vector() - self.u_data.vector())[self.penalty_dofs]
        return 2. * self.penalty_weight.values()[0] * du
    
    def build_variational_functionals(self, f, integ_degree):
        if f is None:
            if self.dep_dim == 1:
                f = df.Constant(0.0)
            else:
                f = df.Constant(self.dep_dim * (0.0, ))
        
        if integ_degree is None:
            self.dxm = df.dx(self.mesh)
            self.integ_degree = 'default'
        else:
            assert(type(integ_degree) == int)
            self.dxm = self.quad_measure(integration_degree = integ_degree)
            self.integ_degree = integ_degree
            df.parameters["form_compiler"]["quadrature_degree"] = self.integ_degree
        
        self.discretize()
        
        ## Concentrated forces using Dirac-delta approach.
        _radius = self.mesh.rmax()
        if self.dep_dim==1:
            self.fs_concentrated = 0. # default
            if len(self.concentrated_forces)>0:
                cfs = self.concentrated_forces['x']
                if len(cfs)>0:
                    self.fs_concentrated = DiracDeltaExpression(radius=_radius, geo_dim=self.geo_dim)
                    for cf in cfs:
                        _correction = self._correction_of_DiracDelta_force(cf['location'], _radius)
                        self.fs_concentrated.add_delta(location=cf['location'], value=cf['value'] \
                                                    , scale=cf['scale']*_correction)
        else:
            _f = self.dep_dim * [0.]
            for direction, cfs in self.concentrated_forces.items():
                if len(cfs)>0:
                    fs = DiracDeltaExpression(radius=_radius, geo_dim=self.geo_dim)
                    for cf in cfs:
                        _correction = self._correction_of_DiracDelta_force(cf['location'], _radius)
                        fs.add_delta(location=cf['location'], value=cf['value'] \
                                     , scale=cf['scale']*_correction)
                    _dir = {'x': 0, 'y': 1, 'z': 2}[direction.lower()]
                    _f[_dir] = fs
            self.fs_concentrated = df.as_vector(tuple(_f))
        
        return f
    
    def discretize(self):
        pass
    
    def build_solver(self, time_varying_loadings=[]):
        self.time_varying_loadings = time_varying_loadings
        if len(self.bcs_DR + self.bcs_DR_inhom) == 0:
            raise ValueError('No boundary conditions have been set to the FEniCS problem.')
    
    def solve(self, t):
        if not hasattr(self, 'solver'):
            raise RuntimeError("A solver must be first set to the FEniCS problem (after setting BCs). Please do it by calling self.build_solver().")
        for l in self.time_varying_loadings:
            l.t = t
        
    def _todo_after_convergence(self, verify_free_residual=False):
        # IMPORTANT: This method MUST be called only if the convergence is met)
        self._assemble_K_t()
        if verify_free_residual:
            # Check if the residual at free DOFs is close to zero (according to the absolute tolerance of the solver)
            res_abs_threshold = 1e-8 # an acceptable threshold
            all_bcs_dofs = self.bcs_DR_dofs + self.bcs_DR_inhom_dofs
            free_dofs = [i for i in range(self.get_i_full().dim()) if i not in all_bcs_dofs]
            res_free = df.assemble(self.get_F_and_u()[0]).get_local()[free_dofs]
            err = np.linalg.norm(res_free)
            if err > res_abs_threshold:
                print(f"\n--- WARNING (CONVERGENCE VERIFIED) ---\n\tThe absolute tolerance reached is {err:.2e} .")
    
    def get_i_full(self):
        # to get the full function space of the problem.
        pass
    
    def get_F_and_u(self):
        # To get the total variational functional (F=0) and its main unknown function
        pass
    
    def get_uu(self, _deepcopy=True):
        # To get the "plottable" function of displacement field (u_u).
        pass
    
    def get_iu(self, _collapse=False):
        # To get the displacement field's function-space (i_u).
        pass
    
    def reaction_force(self, bc_measure): ### !!!! has inaccuracy !!!
        n = df.FacetNormal(self.mesh)
        return df.assemble(df.dot(df.dot(self.sig_u, n), n) * bc_measure)
    
    def reset_fields(self):
        pass
        
    def quad_measure(self, integration_degree=1, domain=None):
        if domain is None:
            domain = self.mesh
        md = {'quadrature_degree': integration_degree, 'quadrature_scheme': 'default'}
        return df.dx(domain = domain, metadata = md)
    
    def _set_K_tangential(self):
        _F, _u = self.get_F_and_u()
        self.K_t_form = df.derivative(_F, _u)

    def revise_BCs(self, remove=False, new_BCs=[], _as='hom'):
        """
        Treats original BCs of the fenics problem object.
        """
        if remove:
            self.bcs_DR = []
            self.bcs_DR_dofs = []
            self.bcs_DR_inhom = []
            self.bcs_DR_inhom_dofs = []
        for bc in new_BCs:
            if _as=='hom': # homogenous
                self.bcs_DR.append(bc)
                self.bcs_DR_dofs.extend([key for key in bc.get_boundary_values().keys()])
            else:
                self.bcs_DR_inhom.append(bc)
                self.bcs_DR_inhom_dofs.extend([key for key in bc.get_boundary_values().keys()])
    
    def _penalize_F(self, F):
        if len(self.penalty_dofs) > 0:
            F[self.penalty_dofs] += self.penalty_forces
        else:
            pass
    
    def _penalize_K_t(self, K_t):
        if len(self.penalty_dofs) > 0:
            if not hasattr(self, '_d_K_t_diag'):
                self._d_K_t_diag = df.PETScVector(self.mesh.mpi_comm(), self.get_i_full().dim())
            K_t.get_diagonal(self._d_K_t_diag)
            K_t.set_diagonal(self._d_K_t_diag + self._K_t_penalty)
        else:
            pass
    
    def _assemble_K_t(self):
        self.K_t = df.assemble(self.K_t_form)
        self._penalize_K_t(self.K_t)
    
    def _correction_of_DiracDelta_force(self, location, radius):
        """
        This returns a correcation coefficent that should scale a concentrated force (a scalar)
        applied at 'location' through a DiracDelta function with 'radius'.
        --> See feniQS.fenics_helpers.fenics_expressions.DiracDeltaExpression class and how that
            is used in FenicsProblem class.
        """
        fs = DiracDeltaExpression(radius=radius, geo_dim=self.geo_dim)
        fs.add_delta(location=location, value=1.)
        if self.dep_dim==1:
            _f = fs
        else:
            _ff = self.dep_dim * [0.]
            _ff[0] = fs
            _f = df.as_vector(tuple(_ff))
        bb = df.assemble(df.inner(_f, self.v_u) * self.dxm)
        _correction = 1. / sum(bb)
        return _correction

class FenicsElastic(FenicsProblem):
    def __init__(self, mat, mesh, fen_config, dep_dim=None \
                 , penalty_dofs=[], penalty_weight=df.Constant(0.)):
        FenicsProblem.__init__(self, mat, mesh, fen_config, dep_dim \
                               , penalty_dofs=penalty_dofs, penalty_weight=penalty_weight)
        
    def build_variational_functionals(self, f=None, integ_degree=None, expr_sigma_scale=1):
        f = FenicsProblem.build_variational_functionals(self, f, integ_degree)  # includes discritization
        
        self.sig_u = expr_sigma_scale * self.mat.sigma(self.u_u)
        eps_v = epsilon(self.v_u, _dim=self.dep_dim)
        a_u = df.inner(self.sig_u, eps_v) * self.dxm
        self.L_u = df.inner(f, self.v_u) * self.dxm
        ## add Neumann BC. terms (if any exist)
        for i, t in enumerate(self.bcs_NM_tractions):
            self.L_u += df.dot(t, self.v_u) * self.bcs_NM_measures[i]
        self.F_u = a_u - self.L_u
        self._set_K_tangential() # The K_tangential is not affected by fs_concentrated, so, is set before adding fs_concentrated.
        ## add concentrated forces
        self.L_u += df.inner(self.fs_concentrated, self.v_u) * self.dxm
        self.F_u = a_u - self.L_u
        self._assemble_K_t()
    
    def discretize(self):
        if self.dep_dim == 1:
            elem_u = df.FiniteElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u)
        else:
            elem_u = df.VectorElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u, dim=self.dep_dim)
        # Define interpolation (shape function)
        self.i_u = df.FunctionSpace(self.mesh, elem_u)
        # Define functions
        self.u_u = df.Function(self.i_u, name='Disp. field')
        self.v_u = df.TestFunction(self.i_u)
        
    def build_solver(self, solver_options=None, time_varying_loadings=[]):
        FenicsProblem.build_solver(self, time_varying_loadings)
        if solver_options is None:
            solver_options = get_fenicsSolverOptions()
        self.solver_options = solver_options
        if self.solver_options['lin_sol']=="direct" or self.solver_options['lin_sol']=="default":
            problem = df.LinearVariationalProblem(a=self.K_t_form, L=self.L_u, u=self.u_u, bcs=self.bcs_DR + self.bcs_DR_inhom)
            self.solver = df.LinearVariationalSolver(problem)
            self.solver.parameters['krylov_solver']["maximum_iterations"] = solver_options['max_iters']
            self.solver.parameters['krylov_solver']["absolute_tolerance"]   = self.solver_options['tol_abs']
            self.solver.parameters['krylov_solver']["relative_tolerance"]   = self.solver_options['tol_rel']
            # self.solver.parameters['lu_solver']["verbose"] = True
                        
        elif self.solver_options['lin_sol']=="iterative":
            if not (df.has_krylov_solver_method(self.solver_options['krylov_method']) or
                    df.has_krylov_solver_preconditioner(self.solver_options['krylov_precon'])):
                raise ValueError(f"Invalid parameters are given for krylov solver.")
            self.solver = MyKrylovSolver(method  = solver_options['krylov_method'],
                                         precond = solver_options['krylov_precon'],
                                         tol_a   = solver_options['tol_abs'],
                                         tol_r   = solver_options['tol_rel'],
                                         max_iter= solver_options['max_iters'])    
    def solve(self, t):
        FenicsProblem.solve(self, t)
        if self.solver_options['lin_sol']=="direct" or self.solver_options['lin_sol']=="default":
            self.solver.solve()
            conv = True
            _it = 0
        elif self.solver_options['lin_sol']=="iterative":
            try:
                self.solver.assemble(self.K_t_form, self.L_u, self.bcs_DR + self.bcs_DR_inhom)
                self.solver(self.u_u)
                conv = True; _it = -1
            except:
                conv = False; _it = -10000
        if conv:
            self._todo_after_convergence()
        return (_it, conv)
    
    def get_i_full(self):
        return self.i_u
    
    def get_F_and_u(self):
        return self.F_u, self.u_u
    
    def get_uu(self):
        return self.u_u
    
    def get_iu(self, _collapse=True):
        return self.i_u
    
    def reset_fields(self, u0=0.0):
        # u0 can be a vector of the same length as self.u_u
        self.u_u.vector()[:] = u0

class FenicsGradientDamage(FenicsProblem, df.NonlinearProblem):
    import warnings
    from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
    warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

    def __init__(self, mat, mesh, fen_config, dep_dim=None \
                 , penalty_dofs=[], penalty_weight=df.Constant(0.) \
                     , K_current=df.Constant(0.0), jac_prep=False):
        assert isinstance(mat, GradientDamageConstitutive)
        ## Check appropriate value for c_min of the material in regards to mesh size:
        b = GradientDamageConstitutive.check_c_min(c_min=mat.c_min, mesh=mesh)
        if not b:
            raise ValueError(f"The minimum value for the gradient damage characteristic length (c_min) is too small compared to the mesh size.\nEither increase this value or refine the mesh.")
        FenicsProblem.__init__(self, mat, mesh, fen_config, dep_dim \
                               , penalty_dofs=penalty_dofs, penalty_weight=penalty_weight)
        self.shF_degree_ebar = fen_config.shF_degree_ebar
        self.K_current = K_current
        self.jac_prep = jac_prep # A flag for preparation of analytical computations of jacobian (w.r.t. either a parameter or displacement)
        self.dF_dps = {} # dictionary of derivatives of self.F (total) w.r.t. parameters (keys are the names of parameters)
        self.dF_dps_vals = {} # dictionary of values (np.array) of derivatives of self.F (total) w.r.t. parameters (keys are the names of parameters)
        df.NonlinearProblem.__init__(self)
        
    def build_variational_functionals(self, f=None, integ_degree=None, expr_sigma_scale=1 \
                                      , hist_storage='quadrature', ebar_residual_weight=1.0):
        self.expr_sigma_scale = expr_sigma_scale
        self.hist_storage = hist_storage
        self.ebar_residual_weight = df.Constant((ebar_residual_weight))
            # To make residual vector of nonlocal-field (ebar) comparable to residual vetor of displacement field.
        
        if self.hist_storage=='quadrature':
            if integ_degree is None:
                # In this case, the 'default' integration degree (see the superclass) might cause inconsistency
                # , since it can internally allocate a different number of integration points than the space
                # behind internal variables ('self.i_K' as will be specified in self.discretize). So, we have to
                # specify a certain value for integ_degree, which is then used - in a consistent way - for both 
                # defining 'self.i_K' and performing the integration.
                integ_degree = max(self.shF_degree_u, self.shF_degree_ebar) + 1
        
        f = FenicsProblem.build_variational_functionals(self, f, integ_degree) # includes discritization
        
        ### for the first field
        u_Kmax = GradientDamageConstitutive.update_K(self.u_K_current, self.u_ebar) # based on the second field (u_ebar)
        self.sig_u = expr_sigma_scale * self.mat.sigma(self.u_u, u_Kmax)
        eps_v = epsilon(self.v_u, _dim=self.dep_dim)
        a_u = df.inner(self.sig_u, eps_v) * self.dxm
        L_u = df.inner(f, self.v_u) * self.dxm
        ## add Neumann BC. terms (if any exist)
        for i, t in enumerate(self.bcs_NM_tractions):
            L_u += df.dot(t, self.v_u) * self.bcs_NM_measures[i]
        ## add concentrated forces
        L_u += df.inner(self.fs_concentrated, self.v_u) * self.dxm
        
        self.F_u = a_u - L_u
        
        ### for the second field
        c_gdm = self.mat.c_min + self.mat.c
        # ## APPROACH 1: without applying divergence theorem
        # lap = div(grad(self.u_ebar))
        # q = self.u_ebar - c_gdm * lap
        # a_ebar = inner(q, self.v_ebar) * self.dxm
        # normal_vector = FacetNormal(self.mesh)
        # a_ebar += c_gdm * inner(dot(grad(self.u_ebar), normal_vector), self.v_ebar) * ds(self.mesh) # For natural boundary conditions
        # eps_eq = self.mat.epsilon_eq(epsilon(self.u_u, _dim=self.dep_dim))
        # L_ebar = inner(eps_eq, self.v_ebar) * self.dxm # Here "eps_eq" is similar to an external force applying to the second field system
        # self.F_ebar = a_ebar - L_ebar
        ## APPROACH 2: after applying divergence theorem
        a_ebar = df.inner(self.u_ebar, self.v_ebar) * self.dxm
        interaction = self.mat.interaction_function(self.mat.gK.g(u_Kmax))
        a_ebar += c_gdm * interaction * df.dot(grad(self.u_ebar), grad(self.v_ebar)) * self.dxm
        eps_eq = self.mat.epsilon_eq(epsilon(self.u_u, _dim=self.dep_dim))
        L_ebar = df.inner(eps_eq, self.v_ebar) * self.dxm # Here "eps_eq" is similar to an external force applying to the second field system
        self.F_ebar = self.ebar_residual_weight * (a_ebar - L_ebar)
        
        self._F_total = self.F_u + self.F_ebar
        
        self._set_K_tangential() # it is always needed for bulding the solver
        self._assemble_K_t()
        
        if self.jac_prep:
            ## Set necessary FEniCS forms and initiate their corresponding evaluations as np.array objects
            self._set_dF_dK0(u_Kmax, expr_sigma_scale) # includes initiation of self.dF_dK0
            self.dKmax_dK0 = np.zeros(self.i_K.dim()) # initiation
            self._set_dKmax_du(u_Kmax) # includes initiation of self.dKmax_du
            self.last_K0 = self.u_K_current.copy(deepcopy=True) # only needed for test purposes
        
    def discretize(self):
        if self.dep_dim == 1:
            elem_u = df.FiniteElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u)
        else:
            elem_u = df.VectorElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u, dim=self.dep_dim)
        elem_ebar = df.FiniteElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_ebar)
        elem_mix = df.MixedElement([elem_u, elem_ebar])
        
        # Define interpolations (shape functions)
        self.i_mix = df.FunctionSpace(self.mesh, elem_mix)
        
        # Define functions
        self.u_mix = df.Function(self.i_mix, name='Main fields')
        self.u_u, self.u_ebar = df.split(self.u_mix)
        
        self.v_mix  = df.TestFunction(self.i_mix)
        self.v_u, self.v_ebar = df.split(self.v_mix)
        
        # "u_K_current" is for internal variables
        if self.hist_storage=='quadrature':
            e_K = df.FiniteElement(family="Quadrature", cell=self.mesh.ufl_cell(), \
                                   degree=self.integ_degree, quad_scheme="default")
            self.i_K = df.FunctionSpace(self.mesh, e_K)
        else: # uses the same interpolation as for "ebar"
            self.i_K = self.i_mix.sub(1).collapse() # "collapse()" is needed for operations related to projection/interpolation.
        self.u_K_current = df.interpolate(self.K_current, self.i_K)
    
    def build_solver(self, solver_options=None, time_varying_loadings=[]):
        FenicsProblem.build_solver(self, time_varying_loadings)
        res , _ = self.get_F_and_u()
        self.assembler = df.SystemAssembler(self.K_t_form, res, self.bcs_DR + self.bcs_DR_inhom)
        if solver_options is None:
            solver_options = get_fenicsSolverOptions()
        self.solver = get_nonlinear_solver(solver_options=solver_options, mpi_comm=self.mesh.mpi_comm())
    
    def F(self, b, x): # Overwrite 'F' method of df.NonlinearProblem
        self.assembler.assemble(b, x)
        self._penalize_F(b)

    def J(self, A, x): # Overwrite 'J' method of df.NonlinearProblem
        self.assembler.assemble(A)
        self._penalize_K_t(A)
    
    def solve(self, t):
        FenicsProblem.solve(self, t)
        _ , u = self.get_F_and_u()
        it, conv = self.solver.solve(self, u.vector())
        if conv:
            self._todo_after_convergence()
        return (it, conv)
    
    def get_i_full(self):
        return self.i_mix
    
    def get_F_and_u(self):
        return self._F_total, self.u_mix
        # return self.F_u + self.F_ebar, self.u_mix
    
    def get_uu(self, _deepcopy=True):
        return self.u_mix.split(deepcopy=_deepcopy)[0]
    
    def get_iu(self, _collapse=False):
        iu = self.i_mix.sub(0)
        if _collapse:
            iu = iu.collapse()
        return iu

    def _todo_after_convergence(self):
        FenicsProblem._todo_after_convergence(self)
        if self.jac_prep:
            ##### assemble other derivative quantities BEFORE updating internal variables
            self._assign_dF_dK0()
            self._assign_dKmax_dK0()
            self._assign_dKmax_du()
            ## assemble (evaluate) all partial derivatives w.r.t. parameters specified by having called "self.set_dF_dp"
            for _name in self.dF_dps.keys():
                self._assemble_dF_dp(_name)
            ## the following is only needed for test purposes
            self.last_K0.vector()[:] = self.u_K_current.vector()[:] # Old Kappa in last time-step: is useful for testing dF_du (d: partial derivative)
                # Reason: F is considered as function of p (parameters), u (solution field) and K0 (Kappa_old at last time-step).
                # So, when taking partial derivative w.r.t. u, we must set self.u_K_current=last_K0.
                # We have "self.u_K_current=last_K0" up until updating self.u_K_current (which is done just next)
        
        ##### updating internal variables (Kappa)
        ### WAY 1: entry-wise maximization (given from Thomas on: https://git.bam.de/mechanics/ttitsche/fenics_snippets/-/blob/master/gdm/gdm.py#L152)
        k = self.u_K_current.vector().get_local()
        if self.hist_storage=='quadrature':
            ebar = self.u_mix.split(deepcopy=True)[1]
            
            # WAY-1-1: Interpolation to self.i_K space
            e = df.interpolate(ebar, self.i_K).vector().get_local()

            # WAY-1-2: Projection to self.i_K space
            # if self.integ_degree>1:
            #     df.parameters["form_compiler"]["representation"] = "quadrature"
            # e = df.project(ebar, self.i_K).vector().get_local()
            # df.parameters["form_compiler"]["representation"] = "uflacs"
        else:
            ebar = self.u_mix.split(deepcopy=True)[1] # deepcopy is necessary in order to get the function's vector() only over the subspace dimension
            e = ebar.vector().get_local()
        self.u_K_current.vector().set_local(np.maximum(k, e))
        
        ### WAY 2: projection/interpolation based on "new K" as a "ufl" object
        # u_K_current_new = GradientDamageConstitutive.update_K(self.u_K_current, self.u_ebar) # based on the second field (u_ebar)
        #   # For being called in the following "assign" method, we must project it first :
        # if self.hist_storage=='quadrature':
        #     if self.integ_degree>1:
        #         df.parameters["form_compiler"]["representation"] = "quadrature"
        #     u_K_current_new = df.project(u_K_current_new, self.i_K, \
        #                                    form_compiler_parameters={"quadrature_degree":self.integ_degree})
        #     df.parameters["form_compiler"]["representation"] = "uflacs"
        # else:
        #     u_K_current_new = df.project(u_K_current_new, self.i_K)
        # self.u_K_current.vector().set_local(u_K_current_new.vector().get_local())
    
    def reset_fields(self, u0=None, ebar0=None, K0=0.0):
        # u0 and ebar0 can be vectors of the same length as self.u_u and self.u_ebar
        if u0 is None and ebar0 is None: # reset to zero
            self.u_mix.vector()[:] = 0 ## faster
        else: # reset to some nonzero values
            if u0 is None:
                u0 = 0
            if ebar0 is None:
                ebar0 = 0
            u_dofs = self.i_mix.sub(0).dofmap().dofs()
            ebar_dofs = self.i_mix.sub(1).dofmap().dofs()
            self.u_mix.vector()[u_dofs] = u0
            self.u_mix.vector()[ebar_dofs] = ebar0
        self.u_K_current.vector()[:] = K0
    
    def set_dF_dp(self, par, _name):
        """
        Given "_name" for a new dictionary key, it sets a FEniCS-form to "self.dF_dps" and its evaluation to "self.dF_dps_vals"
        , which are representing:
            the jacobian of the residual; i.e. "self.fen.get_F_and_u()[0]"; w.r.t. the given "par".
        "par" must have been already "softenned" as a FEniCS function or variable.
        """
        if _name=='penalty_weight':
            self.dF_dps[_name] = None # Not possible to express it in the form of a FEniCS function/constant.
        else:
            self.dF_dps[_name] = df.diff(self.get_F_and_u()[0], par)
        # self.dF_dps_vals[_name] = df.assemble(val).get_local() # evaluated as np.array
        self._assemble_dF_dp(_name)
    
    def _assemble_dF_dp(self, _name):
        if _name=='penalty_weight':
            self.dF_dps_vals[_name] = self._K_t_penalty.get_local()
        else:
            self.dF_dps_vals[_name] = df.assemble(self.dF_dps[_name]).get_local() # evaluated as np.array
        ### IMPORTANT:
            # Due to self.dF_dps_vals being a dictionary, any reference to it (somewhere else) will also be updated
            # as soon as its values are updated here.
    
    def get_dF_dp(self, _name, _eval=True):
        if _eval:
            return self.dF_dps_vals[_name]
        else:
            return self.dF_dps[_name]
    
    def _assign_dF_dK0(self):
        if self.integ_degree>1:
            df.parameters["form_compiler"]["representation"] = "quadrature"
        self.dF_dK0 = df.assemble(self.dF_dK0_form).array()
        df.parameters["form_compiler"]["representation"] = "uflacs"
        
    def _set_dF_dK0(self, u_Kmax, expr_sigma_scale):
        ### establish dF_dK0_form (w.r.t. K_old) (d: partial derivative)
        
        ## WAY 1)
        dF_dsigma = epsilon(self.v_u, _dim=self.dep_dim)
        dsigma_dD = expr_sigma_scale * self.mat.d_sigma_d_D(self.u_u, u_Kmax)
        dD_dKmax = self.mat.gK.dg_dK(u_Kmax)
        dKmax_dK0 = GradientDamageConstitutive.d_K_d_K_old(self.u_K_current, self.u_ebar)
        K_trial = df.TrialFunction(self.i_K)
        self.dF_dK0_form = df.inner(dF_dsigma, dsigma_dD * dD_dKmax * dKmax_dK0 * K_trial) * self.dxm
            # OR:
        # self.dF_dK0_form = df.inner(dD_dKmax * dKmax_dK0 * K_trial * dF_dsigma, dsigma_dD) * self.dxm
        
        ## WAY 2) (only working for integ_degree=1)
        # self.dF_dK0_form = df.derivative(self.get_F_and_u()[0], self.u_K_current)
        
        ## initiate the assembled "self.dF_dK0"
        self._assign_dF_dK0()
        
    def _assign_dKmax_dK0(self):
        """
        Note: The resulting "self.dKmax_dK0" will become a vector, which is the diagonal and only nonzero terms of the actual matrix of dKmax_dK0.
        """
        dKmax_dK0 = GradientDamageConstitutive.d_K_d_K_old(self.u_K_current, self.u_ebar)
        ## The result must be projected (--------- can be replaced with LocalProjector -----------)
        if self.integ_degree>1:
            df.parameters["form_compiler"]["representation"] = "quadrature"
        dKmax_dK0 = df.project(dKmax_dK0, self.i_K)
        self.dKmax_dK0[:] = dKmax_dK0.vector().get_local()
        df.parameters["form_compiler"]["representation"] = "uflacs"
        
    def _set_dKmax_du(self, u_Kmax):
        ## Required functions for computing derivatives
        uK_ = df.TestFunction(self.i_K)
        du = df.TrialFunction(self.i_mix)
        ## WAY 1:
        self.dKmax_du_form = df.derivative(uK_ * u_Kmax * self.dxm, self.u_mix, du)
        ## WAY 2: (gives ERROR due to self.u_mix being mixed function)
        # self.dKmax_du_form = uK_ * df.diff(u_Kmax, self.u_mix) * du * self.dxm
        
        ## Scaling needed in "_assign_dKmax_du" method
        if self.integ_degree>1:
            df.parameters["form_compiler"]["representation"] = "quadrature"
        self.K_volumes = df.assemble(uK_ * self.dxm).get_local() # volume considered by each integration point
        self.dKmax_du = df.assemble(self.dKmax_du_form).array() # initiate self.dKmax_du
        df.parameters["form_compiler"]["representation"] = "uflacs"
    
    def _assign_dKmax_du(self):
        if self.integ_degree>1:
            df.parameters["form_compiler"]["representation"] = "quadrature"
        self.dKmax_du[:][:] = df.assemble(self.dKmax_du_form).array()
        df.parameters["form_compiler"]["representation"] = "uflacs"
        ## Scaling to compensate the effect of mesh sizes
        self.dKmax_du = self.dKmax_du / self.K_volumes[:, np.newaxis] # every i-th row is divided by self.K_volumes[i]

class GDM_DOFs:
    def __init__(self, fen):
        assert(type(fen) is FenicsGradientDamage)
        self.fen = fen
        self.u = self.fen.i_mix.sub(0).dofmap().dofs()
        self.e = self.fen.i_mix.sub(1).dofmap().dofs()
        DR = self.fen.bcs_DR_dofs + self.fen.bcs_DR_inhom_dofs
        self.free = [i for i in self.fen.i_mix.dofmap().dofs() if i not in DR]    
        
        self.fix = []
        self.impose = []
        self.load = []
    
    def fix_at(self, points, i, _add=False): # i is the sub-space on which the points should be fixed
        if _add:
            self.fix = self.fix + dofs_at(points, self.fen.i_mix, i)
        else:
            self.fix = dofs_at(points, self.fen.i_mix, i)
        self.unfix = [d for d in self.fen.i_mix.dofmap().dofs() if d not in self.fix]
    
    def impose_at(self, points, i, _add=False): # i is the sub-space on which the points should be imposed
        new_dofs = dofs_at(points, self.fen.i_mix, i)
        if _add:
            self.impose = self.impose + new_dofs
        else:
            self.impose = new_dofs
        return new_dofs
    
    def load_at(self, points, i, _add=False): # i is the sub-space on which the points should be loaded
        if _add:
            self.load = self.load + dofs_at(points, self.fen.i_mix, i)
        else:
            self.load = dofs_at(points, self.fen.i_mix, i)

class Residual_GDM:
    def __init__(self, fen, res_dofs=None, dF_dps=None):
        """
        This calss provides a callable suitable for quantities related to "residuals" of a GDM problem.
        These quantities are requaired when considering "residuals" as a forward-model's output.
        These quantities are:
            g: forward-model output (residuals)
            dg_dps: derivative of g w.r.t. parameters which are considered in self.dF_dps (explained below)
            dg_du: derivative of g w.r.t. u
            dg_dK: derivative of g w.r.t. Kappa
        , where:            
            self.dF_dps: dictionary of evaluated derivatives of total "F" of self.fen with respect to desired parameters.
            IMPORTANT: It is assumed that self.dF_dps will always be updated according to the last solution at any time (t).
        """
        assert type(fen)==FenicsGradientDamage
        self.fen = fen
        if dF_dps is None:
            dF_dps = self.fen.dF_dps_vals
        self.dF_dps = dF_dps
        self.pars_keys = self.dF_dps.keys() # <class 'dict_keys'>
        
        if res_dofs is None:
            print('WARNING: No residual dofs have been given. By default, all displacement dofs are considered.')
            res_dofs = self.fen.i_mix.sub(0).dofmap().dofs()
        self.res_dofs = res_dofs    
    
    def __call__(self, t):
        F, _ = self.fen.get_F_and_u()        
        ### VERSION-1)  "g" considered as function of u, p, Kappa_old
        g = df.assemble(F).get_local()[self.res_dofs]
        dg_dps = {} # a dictionary
        for _k, _d in self.dF_dps.items():
            dg_dps[_k] = _d[self.res_dofs] # Here, "dF_dp" is identical to "dg_dp", since "F" is identical to "g".
        dg_du = self.fen.K_t.array()[self.res_dofs, :] # with Kappa_old
        dg_dK = self.fen.dF_dK0[self.res_dofs, :] # w.r.t. Kappa_old
        return g, dg_dps, dg_du, dg_dK
    