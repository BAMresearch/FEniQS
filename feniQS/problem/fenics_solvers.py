import sys
if './' not in sys.path:
    sys.path.append('./')

import dolfin as df

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