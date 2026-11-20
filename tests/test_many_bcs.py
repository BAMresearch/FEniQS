import sys
if './' not in sys.path:
    sys.path.append('./')

import unittest
from feniQS.fenics_helpers.fenics_functions import *

tol = 1e-10
deg1 = 3
deg2 = 1
res = 3
shuffle_points = True # True: points given to ManyBCs are in a random order (as DVC points are), not in the order of mesh vertices

fraction_of_points = 0.4 # fraction (<1) of all mesh vertices which are loaded by BCs in the tests of a partial set of points (the rest stay free)

def get_points(mesh, fraction=1):
    """
    The mesh vertices at which BCs are applied: a random 'fraction' of all vertices (all of them by default).
    They are in a random order if 'shuffle_points', otherwise in the order of vertices.
    """
    points = mesh.coordinates()
    n = max(1, int(round(fraction * len(points))))
    ids = np.random.default_rng(0).permutation(len(points))[:n]
    if not shuffle_points:
        ids = np.sort(ids)
    return points[ids]

class MixBase:
    def check_response_at_bc_points(self, vals):
        """
        Checks the solved displacement-like field 'u1' at the points where BCs were applied (self.points):
        its leading components must be equal to the given values (vals[i] is a scalar/array for self.points[i]) and
        the remaining components (not loaded by the BC) must have remained ZERO.
        """
        u1_eval = self.u_mix.split()[0]
        for i, p in enumerate(self.points):
            v = np.atleast_1d(vals[i])
            u = u1_eval(p)
            assert(np.linalg.norm(u[:len(v)] - v) < tol)
            assert(np.linalg.norm(u[len(v):]) < tol)

class Mix2D(MixBase):
    def __init__(self, fraction=1):
        self.mesh = df.RectangleMesh(df.Point(0,0), df.Point(2,1), 2*res, 1*res)
        elem_1 = df.VectorElement('CG', self.mesh.ufl_cell(), deg1, dim=2)
        elem_2 = df.FiniteElement('CG', self.mesh.ufl_cell(), deg2)
        elem_mix = df.MixedElement([elem_1, elem_2])
        # Define interpolations (shape functions)
        self.i_mix = df.FunctionSpace(self.mesh, elem_mix)
        self.i1 = self.i_mix.sub(0)
        self.i2 = self.i_mix.sub(1)
        # Define functions
        self.u_mix = df.Function(self.i_mix, name='Mixed Fields')
        self.u1, self.u2 = df.split(self.u_mix)
        self.points = get_points(self.mesh, fraction)
        self.vals = []
        for i in range(len(self.points)): # set some arbitrary values
            self.vals.append(np.array([2*i+1, 2*i+2]))
    
    def solve(self, bc):
        """ A dummy problem of:
            sigma(self.u1) = 1 * grad(self.u1) = 0
            self.u2 = sum(self.u1)
        """
        v1, v2 = df.TestFunctions(self.i_mix)
        u11, u12 = df.split(self.u1)
        a1 = df.inner(1 * df.grad(self.u1), df.grad(v1)) * df.dx
        a2 = df.inner((self.u2 - u11 - u12), v2) * df.dx
        df.solve(a1+a2==0, self.u_mix, bc)
        
class Mix3D(MixBase):
    def __init__(self, fraction=1):
        self.mesh = df.BoxMesh(df.Point(0,0,0), df.Point(3,2,1), 3*res, 2*res, 1*res)
        elem_1 = df.VectorElement('CG', self.mesh.ufl_cell(), deg1, dim=3)
        elem_2 = df.FiniteElement('CG', self.mesh.ufl_cell(), deg2)
        elem_mix = df.MixedElement([elem_1, elem_2])
        # Define interpolations (shape functions)
        self.i_mix = df.FunctionSpace(self.mesh, elem_mix)
        self.i1 = self.i_mix.sub(0)
        self.i2 = self.i_mix.sub(1)
        # Define functions
        self.u_mix = df.Function(self.i_mix, name='Mixed Fields')
        self.u1, self.u2 = df.split(self.u_mix)
        self.points = get_points(self.mesh, fraction)
        self.vals = []
        for i in range(len(self.points)): # set some arbitrary values
            self.vals.append(np.array([3*i+1, 3*i+2, 3*i+3]))
    
    def solve(self, bc):
        """ A dummy problem of:
            sigma(self.u1) = 1 * grad(self.u1) = 0
            self.u2 = sum(self.u1)
        """
        v1, v2 = df.TestFunctions(self.i_mix)
        u11, u12, u13 = df.split(self.u1)
        a1 = df.inner(1 * df.grad(self.u1), df.grad(v1)) * df.dx
        a2 = df.inner((self.u2 - u11 - u12 - u13), v2) * df.dx
        df.solve(a1+a2==0, self.u_mix, bc)

class TestManyBCs(unittest.TestCase):
    
    def test_mix2d_bc2d(self): # apply bc also in 2-D
        mix = Mix2D()
        bcs = ManyBCs(mix.i_mix, mix.i1, mix.points)
        bcs.assign(mix.vals)
        mix.solve(bcs.bc)
        u1_eval = mix.u_mix.split()[0]
        for i, p in enumerate(mix.points):
            assert(np.linalg.norm(u1_eval(p) - mix.vals[i]) < tol)
    
    def test_mix2d_bc2d_repeated_assign(self): # 'assign' several times: the BC must follow the latest values (e.g. load steps)
        mix = Mix2D()
        bcs = ManyBCs(mix.i_mix, mix.i1, mix.points)
        for shift in [0., 10., -5.]: # a different set of values at every round
            vals = [v + shift for v in mix.vals]
            bcs.assign(vals)
            mix.solve(bcs.bc)
            u1_eval = mix.u_mix.split()[0]
            for i, p in enumerate(mix.points):
                assert(np.linalg.norm(u1_eval(p) - vals[i]) < tol)

    def test_mix2d_bc2d_fraction(self): # BCs only at a fraction of the points; the other DOFs must stay free
        mix = Mix2D(fraction=fraction_of_points)
        assert(len(mix.points) < len(mix.mesh.coordinates()))
        bcs = ManyBCs(mix.i_mix, mix.i1, mix.points)
        assert(len(bcs.dofs) == 2 * len(mix.points)) # exactly 2 constrained DOFs per given point (no more, no less)
        bcs.assign(mix.vals)
        mix.solve(bcs.bc)
        mix.check_response_at_bc_points(mix.vals)

    def test_mix2d_bc1d_fraction(self):
        mix = Mix2D(fraction=fraction_of_points)
        bcs = ManyBCs(mix.i_mix, mix.i1.sub(0), mix.points, sub_ids=[0,0])
        assert(len(bcs.dofs) == len(mix.points))
        vals_1d = [v[0] for v in mix.vals]
        bcs.assign(vals_1d)
        mix.solve(bcs.bc)
        mix.check_response_at_bc_points(vals_1d)

    def test_mix3d_bc3d_fraction(self):
        mix = Mix3D(fraction=fraction_of_points)
        assert(len(mix.points) < len(mix.mesh.coordinates()))
        bcs = ManyBCs(mix.i_mix, mix.i1, mix.points)
        assert(len(bcs.dofs) == 3 * len(mix.points))
        bcs.assign(mix.vals)
        mix.solve(bcs.bc)
        mix.check_response_at_bc_points(mix.vals)

    def test_mix2d_bc1d(self): # apply bc only in 1-D
        mix = Mix2D()
        bcs = ManyBCs(mix.i_mix, mix.i1.sub(0), mix.points, sub_ids=[0,0])
        vals_1d = [v[0] for v in mix.vals]
        bcs.assign(vals_1d)
        mix.solve(bcs.bc)
        u1_eval = mix.u_mix.split()[0]
        for i, p in enumerate(mix.points):
            assert(np.linalg.norm(u1_eval(p)[0] - vals_1d[i]) < tol)
            assert(np.linalg.norm(u1_eval(p)[1]) < tol) # must have remained ZERO
    
    def test_mix3d_bc3d(self): # apply bc also in 3-D
        mix = Mix3D()
        bcs = ManyBCs(mix.i_mix, mix.i1, mix.points)
        bcs.assign(mix.vals)
        mix.solve(bcs.bc)
        u1_eval = mix.u_mix.split()[0]
        for i, p in enumerate(mix.points):
            assert(np.linalg.norm(u1_eval(p) - mix.vals[i]) < tol)
            # print(f"{u1_eval(p)}, {mix.vals[i]}")
    
    def test_mix3d_bc1d(self): # apply bc only in 1-D
        mix = Mix3D()
        bcs = ManyBCs(mix.i_mix, mix.i1.sub(0), mix.points, sub_ids=[0,0])
        vals_1d = [v[0] for v in mix.vals]
        bcs.assign(vals_1d)
        mix.solve(bcs.bc)
        u1_eval = mix.u_mix.split()[0]
        for i, p in enumerate(mix.points):
            assert(np.linalg.norm(u1_eval(p)[0] - vals_1d[i]) < tol)
            assert(np.linalg.norm(u1_eval(p)[1]) < tol) # must have remained ZERO
            assert(np.linalg.norm(u1_eval(p)[2]) < tol) # must have remained ZERO
            # print(f"{u1_eval(p)}, {vals_1d[i]}")

if __name__ == "__main__":
    df.set_log_level(30)
    unittest.main()
