import dolfin as df
from feniQS.general.general import CollectPaths

pth_fenics_solvers = CollectPaths('./feniQS/problem/fenics_solvers.py')

def get_fenicsSolverOptions(case='nonlinear', lin_sol='default'):
    nln_so = None # By default, NO need for nonlinear solver options
    if case.lower()=='nonlinear' or case.lower()=='nln':
        nln_so = get_nonlinear_solver_options()
    
    if lin_sol.lower()=='default' or lin_sol.lower()=='direct':
        if nln_so is None:
            lin_so = get_default_linear_solver_options()
        else:
            lin_so = 'default' # For now: a default linear solver nested in a nonlinear solver cannot adopt any specifications.
    elif lin_sol.lower()=='iterative' or lin_sol.lower()=='krylov':
        lin_so = get_Krylov_solver_options()
    else:
        raise ValueError(f"The given linear solver '{lin_sol}' is not recognized. Possible choices are: 'default', 'direct', 'iterative', 'krylov'.")

    return {
    'nln_sol_options': nln_so,
    'lin_sol_options': lin_so,
    }

def get_default_linear_solver_options():
    # NOT effective when having a nonlinear solver (see 'get_fenicsSolverOptions' above)
    return {
    'type': 'default (direct)',
    'max_iters': 10,
    'allow_nonconvergence_error': False,
    'tol_abs': 1e-10,
    'tol_rel': 1e-10,
    }

def get_Krylov_solver_options():
    """
    Options/parameters regarding a dolfin.KrylovSolver solver.
    """
    return {
    'type': 'iterative (krylov)',
    'max_iters': 10000,
    'allow_nonconvergence_error': False,
    'tol_abs': 1e-10, # NOT effective when having a nonlinear solver (see 'get_nonlinear_solver' below)
    'tol_rel': 1e-8,
    'method': "gmres",
    'precon': "default",
    }

def get_nonlinear_solver_options():
    return {
    'type': "newton",
    'max_iters': 14,
    'allow_nonconvergence_error': False,
    'tol_abs': 1e-10,
    'tol_rel': 1e-10,
    }

def is_default_lin_sol_options(lin_so):
    """
    'lin_so': Any given 'linear solver options'.
    This returns a boolean saying whether the given 'lin_so' regards to 'default' ('direct') solver or not.
    """
    if isinstance(lin_so, str):
        if 'default' in lin_so.lower() or 'direct' in lin_so.lower():
            bb = True
        else:
            raise ValueError(f"The given linear solver options '{lin_so}' is not recognized. Valid options are: 'default', 'direct'.")
    else: # must be dictionary
        if not isinstance(lin_so, dict):
            raise ValueError(f"The given linear solver options must be a dictionary (if not being either 'default' or 'direct').")
        _type = lin_so['type']
        if 'default' in _type.lower() or 'direct' in _type.lower():
            bb = True
        elif 'iterative' in _type.lower() or 'krylov' in _type.lower():
            bb = False
        else:
            raise ValueError(f"The given linear solver options has unrecognized type='{_type}'. Valid types are: 'default', 'direct', 'iterative', 'krylov'.")
    return bb

def get_nonlinear_solver(solver_options, mpi_comm):
    nln_so = solver_options['nln_sol_options']
    lin_so = solver_options['lin_sol_options']
    bb = is_default_lin_sol_options(lin_so)
    
    if 'newton' in nln_so['type'].lower():
        if bb:
            solver = df.NewtonSolver() # With all default options for the linear solver.
            ## ?: how to adjust only tolerances and other options of the underlying LINEAR solver.
        else: # iterative (Krylove)
            assert ('iterative' in lin_so['type'].lower()) or ('krylov' in lin_so['type'].lower())
            method = lin_so['method']

            pc = df.PETScPreconditioner(lin_so['precon'])
            lin_solver = df.PETScKrylovSolver(method, pc)
                ## OR:
            # lin_solver = df.PETScKrylovSolver(method, lin_so['precon'])
            
            lin_solver.parameters['error_on_nonconvergence'] = lin_so['allow_nonconvergence_error']
            lin_solver.parameters['maximum_iterations'] = lin_so['max_iters']
            lin_solver.parameters['relative_tolerance'] = lin_so['tol_rel']
            ## NOTE: The following absolute_tolerance would apparently hinder the convergence of the linear solver!
            # lin_solver.parameters['absolute_tolerance'] = lin_so['tol_abs']
            # lin_solver.parameters['monitor_convergence'] = True
            # lin_solver.parameters['nonzero_initial_guess'] = True
            
            # NOTE: It is not obvious how the matrix 'A' of Newton-iteration is assigned to liner-solver's A matrix.
                  # But it seems that, this is just automatically handled by FEniCS.
            solver = df.NewtonSolver(mpi_comm, lin_solver, df.PETScFactory.instance())
        
    elif 'snes' in nln_so['type'].lower():
        if bb:
            solver = df.PETScSNESSolver() # With all default options for the linear solver.
            ## ?: how to adjust only tolerances and other options of the underlying LINEAR solver.
        else:
            raise NotImplementedError(f"Iterative linear solver together with 'snes' solver is not implemented.")
    else:
        raise KeyError(f"The nonlinear solver type '{nln_so['type']}' is not recognized.")

    solver.parameters["absolute_tolerance"] = nln_so['tol_abs']
    solver.parameters["relative_tolerance"] = nln_so['tol_rel']
    solver.parameters["error_on_nonconvergence"] = nln_so['allow_nonconvergence_error']
    solver.parameters["maximum_iterations"] = nln_so['max_iters']

    return solver

class MyKrylovSolver:
    def __init__ (self, method = 'cg', precond = 'default' \
                  , tol_a = 1e-12, tol_r = 1e-12, max_iter=200):
        assert df.has_krylov_solver_method(method)
        assert df.has_krylov_solver_preconditioner(precond)
        
        pc = df.PETScPreconditioner(precond)
        self.solver = df.PETScKrylovSolver(method, pc)
            ## OR:
        # self.solver = df.PETScKrylovSolver(method, self.precond)

        self.solver.parameters['absolute_tolerance'] = tol_a
        self.solver.parameters['relative_tolerance'] = tol_r
        self.solver.parameters['maximum_iterations'] = max_iter
    
    def __call__(self, u):
        if (hasattr(self, 'A') & hasattr(self, 'b')):
            self.solver.set_operator(self.A)
            self.solver.solve(u.vector(), self.b)
        else:
            print(f"WARNING: Use the assemble() method first before calling the solver!")
          
    def assemble(self, a, L, bcs):
        self.A = df.assemble(a) # lhs matrixA
        self.b = df.assemble(L) # rhs vector
        for bc in bcs:
            bc.zero_columns(self.A, self.b, 1.0)