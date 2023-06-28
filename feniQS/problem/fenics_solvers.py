import dolfin as df
from feniQS.general.general import CollectPaths

pth_fenics_solvers = CollectPaths('./feniQS/problem/fenics_solvers.py')

def get_fenicsSolverOptions():
    return {
    'max_iters': 14,
    'allow_nonconvergence_error': False,
    'type': "newton",
    'tol_abs': 1e-10,
    'tol_rel': 1e-10,
    'lin_sol': 'default', # direct solver
        # The following two are relevant only if lin_sol=='iterative' or 'krylov'
    'krylov_method': "gmres",
    'krylov_precon': "default",
    }

def get_nonlinear_solver(solver_options, mpi_comm):
    allow_nonconvergence_error = solver_options['allow_nonconvergence_error']
    max_iters = solver_options['max_iters']
    s = solver_options['type']
    tol_abs = solver_options['tol_abs']
    tol_rel = solver_options['tol_rel']
    lin_sol = solver_options['lin_sol']
    bb = ('default' in lin_sol.lower()) or ('direct' in lin_sol.lower()) # direct/default solver
    
    if 'newton' in s.lower():
        if bb:
            solver = df.NewtonSolver() # A linear solver can calso be specified as input to that.
        else: # 'iterative'
            method = solver_options['krylov_method']
            pc = solver_options['krylov_precon']
            lin_solver = df.PETScKrylovSolver(method, pc)
            
            lin_solver.parameters['error_on_nonconvergence'] = allow_nonconvergence_error
            lin_solver.parameters['maximum_iterations'] = max_iters
            lin_solver.parameters['relative_tolerance'] = tol_rel
            lin_solver.parameters['absolute_tolerance'] = tol_abs
            # lin_solver.parameters['monitor_convergence'] = True
            # lin_solver.parameters['nonzero_initial_guess'] = True
            
            solver = NewtonSolverFromLinearSolver(mpi=mpi_comm, lin_solver=lin_solver)
            
            ### The following seems to work too,
                # however, it is not obvious how the matrix 'A' of Newton-iteration is assigned to liner-solver's A matrix.
                # Maybe this is just automatically done !?
            # solver = df.NewtonSolver(mpi_comm, lin_solver, df.PETScFactory.instance())
        
    elif 'snes' in s.lower():
        if bb:
            solver = df.PETScSNESSolver()
        else:
            raise NotImplementedError(f"Iterative linear solver together with 'snes' solver is not implemented.")
    else:
        raise KeyError(f"The solver type '{s}' is not recognized.")
    solver.parameters["absolute_tolerance"] = tol_abs
    solver.parameters["relative_tolerance"] = tol_rel
    solver.parameters["error_on_nonconvergence"] = allow_nonconvergence_error
    solver.parameters["maximum_iterations"] = max_iters

    return solver

class NewtonSolverFromLinearSolver(df.NewtonSolver):
    def __init__(self, mpi, lin_solver):
        df.NewtonSolver.__init__(self, mpi, lin_solver, df.PETScFactory.instance())
    
    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

class MyKrylovSolver:
    def __init__ (self, method = 'cg', precond = 'default', tol_a = 1e-12, tol_r = 1e-12, max_iter=200):
        assert df.has_krylov_solver_method(method)
        assert df.has_krylov_solver_preconditioner(precond)
        
        self.method     = method
        self.precond    = precond
        self.tol_a      = tol_a
        self.tol_r      = tol_r
        self.max_iter   = max_iter
        
        self.solver = df.KrylovSolver(self.method, self.precond)
        self.solver.parameters['absolute_tolerance'] = self.tol_a
        self.solver.parameters['relative_tolerance'] = self.tol_r
        self.solver.parameters['maximum_iterations'] = self.max_iter
    
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