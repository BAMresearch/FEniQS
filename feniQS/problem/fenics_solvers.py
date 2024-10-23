import dolfin as df
from feniQS.general.general import CollectPaths

pth_fenics_solvers = CollectPaths('./feniQS/problem/fenics_solvers.py')

"""
IMPORTANT NOTEs (in the implementation of this script):
    NOTE-0:
        An iterative (krylov) linear solver can be nested in a nonlinear solver, so far ONLY
        when the type of that nonlinear solver is 'newton'.
    NOTE-1:
        If the linear solver nested in a nonlinear solver is going to be default (direct),
        that linear solver cannot be customized (does NOT adopt any specifications).
            (Or equivalently:)
        The only customizable linear solver nested in a nonlinear solver is iterative (krylov).
    NOTE-2:
        If the linear solver nested in a nonlinear solver is going to be iterative (krylov),
        the option 'tol_abs' of that linear solver is omitted; otherwise, convergence issues raise.
"""

def remove_ineffective_fenicsSolverOptions(solver_options, _print=True):
    """
    The input 'solver_options' generally is a dictionary with various items, some of which
        may be ineffective, according to the implementation of solvers.
    This method takes a copy of the input 'solver_options' and modifies it by removing
        those ineffective options.
    """
    import copy
    so = copy.deepcopy(solver_options)
    lin_so = so['lin_sol_options']
    nln_so = so['nln_sol_options']
    _msg = 'WARNING (Solver Options):'; len_msg = len(_msg)
    if isinstance(lin_so, str):
        pass
    else:
        _type = lin_so['type']
        if 'default' in _type.lower() or 'direct' in _type.lower():
            if nln_so is None:
                # One may have assigned the following ineffective options, which are now removed.
                for k in ['method', 'precon']:
                    _b = so['lin_sol_options'].pop(k, None)
                    if _b:
                        _msg += f"\n\tThe ineffective option '{k}' was removed from the linear solver options."
            else:
                so['lin_sol_options'] = 'default' # (see NOTE-1 above)
                _msg += f"\n\tThe ineffective linear solver options were all removed and replaced with 'default'."
        elif 'iterative' in _type.lower() or 'krylov' in _type.lower():
            if nln_so is None:
                pass
            else:
                so['lin_sol_options'].pop('tol_abs', None) # (see NOTE-2 above)
                _msg += f"\n\tThe ineffective option 'tol_abs' was removed from the linear solver options."
        else:
            raise ValueError(f"The given linear solver options has unrecognized type='{_type}'. Valid types are: 'default', 'direct', 'iterative', 'krylov'.")
    if _print and len(_msg)>len_msg:
        print(_msg)
    return so

def get_fenicsSolverOptions(case='nonlinear', lin_sol='default'):
    nln_so = None # By default, NO need for nonlinear solver options
    if case.lower()=='nonlinear' or case.lower()=='nln':
        nln_so = get_nonlinear_solver_options()
    
    if lin_sol.lower()=='default' or lin_sol.lower()=='direct':
        if nln_so is None:
            lin_so = get_default_linear_solver_options()
        else:
            lin_so = 'default' # (see NOTE-1 above)
    elif lin_sol.lower()=='iterative' or lin_sol.lower()=='krylov':
        lin_so = get_Krylov_solver_options()
    else:
        raise ValueError(f"The given linear solver '{lin_sol}' is not recognized. Possible choices are: 'default', 'direct', 'iterative', 'krylov'.")

    return {
    'nln_sol_options': nln_so,
    'lin_sol_options': lin_so,
    }

def get_default_linear_solver_options():
    """
    Options/parameters regarding a direct linear solver.
        These are NOT effective when a linear solver is nested within a nonlinear solver (see NOTE-1 above).
    """
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
    'tol_abs': 1e-10,
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
    bb = is_default_lin_sol_options(lin_so) # default (direct) linear solver
    
    if 'newton' in nln_so['type'].lower():
        if bb:
            solver = df.NewtonSolver()
                # (see NOTE-1 above): How to customize the underlying default (direct) linear solver?
        else:
            assert ('iterative' in lin_so['type'].lower()) or ('krylov' in lin_so['type'].lower())
            method = lin_so['method']

            pc = df.PETScPreconditioner(lin_so['precon'])
            lin_solver = df.PETScKrylovSolver(method, pc)
                ## OR:
            # lin_solver = df.PETScKrylovSolver(method, lin_so['precon'])
            
            lin_solver.parameters['error_on_nonconvergence'] = lin_so['allow_nonconvergence_error']
            lin_solver.parameters['maximum_iterations'] = lin_so['max_iters']
            lin_solver.parameters['relative_tolerance'] = lin_so['tol_rel']
            # lin_solver.parameters['absolute_tolerance'] = lin_so['tol_abs'] # Must be excluded (see NOTE-2 above)
            # lin_solver.parameters['monitor_convergence'] = True
            # lin_solver.parameters['nonzero_initial_guess'] = True
            
            # NOTE: It is not obvious how the matrix 'A' of Newton-iteration is assigned to liner-solver's A matrix.
                  # But it seems that, this is just automatically handled by FEniCS.
            solver = df.NewtonSolver(mpi_comm, lin_solver, df.PETScFactory.instance())
        
    elif 'snes' in nln_so['type'].lower():
        if bb:
            solver = df.PETScSNESSolver()
                # (see NOTE-1 above): How to customize the underlying default (direct) linear solver?
        else:
            # (see NOTE-0 above)
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