from feniQS.problem.problem import *

pth_problem_shell = CollectPaths('./feniQS/problem/problem_shell.py')
pth_problem_shell.add_script(pth_problem)
pth_problem_shell.pths.remove(pth_damage.path0)

class ShellFenConfigDefault:
    def __init__(self):
        self.shF_degree_u = 2
        self.shF_degree_theta = 1
        self.el_family = 'Lagrange'

class FenicsElasticShell(FenicsProblem):
    def __init__(self, mat, mesh, thick, fen_config, dep_dim=3 \
                 , penalty_dofs=[], penalty_weight=df.Constant(0.)):
        assert isinstance(mat, ElasticConstitutive)
        if mat.constraint.lower()!='plane_stress':
            raise NotImplementedError("Elastic shell formulation is only implemented for PLANE_STRESS.")
        mat.dim = 3 # to avoid incompatibility
        FenicsProblem.__init__(self, mat, mesh, fen_config, dep_dim \
                               , penalty_dofs=penalty_dofs, penalty_weight=penalty_weight)
        self.shF_degree_theta = fen_config.shF_degree_theta
        self.thick = thick
        
    def build_variational_functionals(self, f=None, integ_degree=None, expr_sigma_scale=1):
        FenicsProblem.build_variational_functionals(self, f, integ_degree) # includes discritization & building external forces
        ## Internal forces
        e1, e2, e3 = local_frame(self.mesh, dim=self.dep_dim, xdmf_file=None)
        P_plane = hstack([e1, e2]) # In-plane projection
        def t_grad(u):
            """Tangential gradient operator"""
            g = df.grad(u)
            return df.dot(g, P_plane)
        t_gu = df.dot(P_plane.T, t_grad(self.u_u))
        eps = df.sym(t_gu)
        kappa = df.sym(df.dot(P_plane.T, t_grad(df.cross(e3, self.u_theta))))
        gamma = t_grad(df.dot(self.u_u, e3)) - df.dot(P_plane.T, df.cross(e3, self.u_theta))
        eps_ = ufl.replace(eps, {self.u_mix: self.v_mix})
        kappa_ = ufl.replace(kappa, {self.u_mix: self.v_mix})
        gamma_ = ufl.replace(gamma, {self.u_mix: self.v_mix})
        def plane_stress_elasticity(e, lamda, mu):
            return lamda * df.tr(e) * df.Identity(2) + 2 * mu * e
        self.N = self.thick * plane_stress_elasticity(eps, self.mat.lamda, self.mat.mu)
        self.M = self.thick ** 3 / 12 * plane_stress_elasticity(kappa, self.mat.lamda, self.mat.mu)
        self.Q = self.mat.mu * self.thick * gamma
        drilling_strain = (t_gu[0, 1] - t_gu[1, 0]) / 2 - df.dot(self.u_theta, e3)
        drilling_strain_ = ufl.replace(drilling_strain, {self.u_mix: self.v_mix})
        self.drilling_stress = self.mat.E * self.thick ** 3 * drilling_strain
        self._F_int = (
            df.inner(self.N, eps_)
            + df.inner(self.M, kappa_)
            + df.dot(self.Q, gamma_)
            + self.drilling_stress * drilling_strain_
        ) * self.dxm
        ## Stiffness matrix
        self._set_K_tangential()
        # self._assemble_K_t()
    
    def get_solution_field(self):
        return self.u_mix

    def discretize(self):
        elem_u = df.VectorElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u, dim=self.dep_dim)
        elem_theta = df.VectorElement("CR", self.mesh.ufl_cell(), self.shF_degree_theta, dim=self.dep_dim)
        elem_mix = df.MixedElement([elem_u, elem_theta])
        self.V = df.FunctionSpace(self.mesh, elem_mix)
        self.i_mix = df.FunctionSpace(self.mesh, elem_mix)
        print(f"Total number of DOFs = {self.i_mix.dim()} .")
    
        self.u_mix = df.Function(self.i_mix, name='Main fields')
        self.u_u, self.u_theta = df.split(self.u_mix)
    
        self.v_mix  = df.TestFunction(self.i_mix)
        self.v_u, self.v_theta = df.split(self.v_mix)
        
    def build_solver(self, solver_options=None, time_varying_loadings=[]):
        FenicsProblem.build_solver(self, time_varying_loadings)
        if solver_options is None:
            solver_options = get_fenicsSolverOptions(case='linear')
        if solver_options['nln_sol_options'] is not None:
            print(f"WARNING: The nonlinear solver options are ignored (set to None) in building the solver of the Elastic problem.")
            solver_options['nln_sol_options'] = None
        lin_so = solver_options['lin_sol_options']
        self.is_default_lin_sol = is_default_lin_sol_options(lin_so)
        if self.is_default_lin_sol:
            problem = df.LinearVariationalProblem(a=self.K_t_form, L=self.get_external_forces() \
                                                  , u=self.get_solution_field(), bcs=self.bcs_DR + self.bcs_DR_inhom)
            self.solver = df.LinearVariationalSolver(problem)
            self.solver.parameters['krylov_solver']["maximum_iterations"] = lin_so['max_iters']
            self.solver.parameters['krylov_solver']["absolute_tolerance"]   = lin_so['tol_abs']
            self.solver.parameters['krylov_solver']["relative_tolerance"]   = lin_so['tol_rel']
            self.solver.parameters['krylov_solver']["error_on_nonconvergence"]   = lin_so['allow_nonconvergence_error']
            # self.solver.parameters['lu_solver']["verbose"] = True
        else:
            if not (df.has_krylov_solver_method(lin_so['method']) or
                    df.has_krylov_solver_preconditioner(lin_so['precon'])):
                raise ValueError(f"Invalid parameters are given for krylov solver.")
            self.solver = MyKrylovSolver(method  = lin_so['method'],
                                         precond = lin_so['precon'],
                                         tol_a   = lin_so['tol_abs'],
                                         tol_r   = lin_so['tol_rel'],
                                         max_iter= lin_so['max_iters'])    
    def solve(self, t):
        FenicsProblem.solve(self, t)
        if self.is_default_lin_sol:
            self.solver.solve()
            conv = True
            _it = 0
        else:
            try:
                self.solver.assemble(self.K_t_form, self.get_external_forces(), self.bcs_DR + self.bcs_DR_inhom)
                self.solver(self.get_solution_field())
                conv = True; _it = -1
            except:
                conv = False; _it = -10000
        if conv:
            self._todo_after_convergence()
        return (_it, conv)
    
    def get_i_full(self):
        return self.i_mix
    
    def get_uu(self, _deepcopy=True):
        return self.u_mix.split(deepcopy=_deepcopy)[0]
    
    def get_iu(self, _collapse=False):
        iu = self.i_mix.sub(0)
        if _collapse:
            iu = iu.collapse()
        return iu
    
    def reset_fields(self, u0=0.0):
        # u0 can be a vector of the same length as self.u_u
        self.u_mix.vector()[:] = u0

def local_frame(mesh, dim=None, xdmf_file='./frame.xdmf'):
    if dim is None:
        dim = mesh.geometric_dimension()
    t = ufl.Jacobian(mesh)
    if dim == 2:
        t1 = df.as_vector([t[0, 0], t[1, 0], 0])
        t2 = df.as_vector([t[0, 1], t[1, 1], 0])
    else:
        t1 = df.as_vector([t[0, 0], t[1, 0], t[2, 0]])
        t2 = df.as_vector([t[0, 1], t[1, 1], t[2, 1]])
    ##### e3 #####
    e3 = df.cross(t1, t2)
    e3 /= df.sqrt(df.dot(e3, e3))
    ##### e1 #####
    e1 = t1 / df.sqrt(df.dot(t1, t1))
    ##### e2 #####
    e2 = df.cross(e3, e1)
    e2 /= df.sqrt(df.dot(e2, e2))
    frame = (e1, e2, e3)
    if xdmf_file!=None:
        with df.XDMFFile(xdmf_file) as efile:
            efile.parameters["functions_share_mesh"] = True
            VT = df.VectorFunctionSpace(mesh, "DG", 0, dim=3)
            for (i, ei) in enumerate(frame):
                fi = df.Function(VT, name="e{}".format(i + 1))
                fi.vector().set_local(df.project(ei, VT).vector().get_local())
                # fi.assign(df.project(frame[i], VT))
                efile.write(fi, 0)
    return frame

def vstack(vectors):
    """Stack a list of vectors vertically."""
    return df.as_matrix([[v[i] for i in range(len(v))] for v in vectors])

def hstack(vectors):
    """Stack a list of vectors horizontally."""
    return vstack(vectors).T