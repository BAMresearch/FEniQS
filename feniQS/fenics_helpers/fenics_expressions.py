import numpy as np
import dolfin as df

from feniQS.general.general import CollectPaths
pth_fenics_expressions = CollectPaths('./feniQS/fenics_helpers/fenics_expressions.py')

class DiracDeltaExpression(df.UserExpression):
    def __init__(self, radius, geo_dim):
        """
        To express a Dirac-delta function at certain (multiple) locations with certain corresponding values.
        A new location/value can be added through the method 'add_delta' (see implementation).
        Each value can be either of:
            float
            df.Expression
            df.UserExpression
        The kernel of the Dirac-delta is from a (multivariate) normal distribution with dimension N = geo_dim (1 or 2 or 3).
        The covariance of such MVN is '_var * np.eye(self.geo_dim)', with '_var = (radius / 2.) ** 2'.
            --> See '_pdf' in the implementation.
        """
        df.UserExpression.__init__(self)
        self.radius = radius
        self._var = (radius / 2.) ** 2
        self.geo_dim = geo_dim
        self.locations = []
        self.values = []
        self.scales = []
    
    def add_delta(self, location=None, value=1., scale=1.):
        if location is None:
            if self.geo_dim==1:
                location = 0.
            else:
                location = self.geo_dim * [0.]
        if self.geo_dim > 1:
            assert len(location)==self.geo_dim
        self.locations.append(location)
        self.values.append(value)
        self.scales.append(scale)
    
    def eval(self, values, x):
        values[0] = 0.
        for il, loc in enumerate(self.locations):
            d = np.array(x) - np.array(loc)
            d2 = sum(d * d)
            _pdf = ((2. * df.pi * self._var) ** (-self.geo_dim / 2.)) * np.exp(- 0.5 * d2 / self._var)
            
            val = self.values[il]
            if isinstance(val, df.Expression):
                val = val(x)
            elif isinstance(val, df.UserExpression):
                vals = [None]
                val.eval(vals, x=x)
                val = vals[0]
            else:
                assert isinstance(val, float) or isinstance(val, int)
            
            values[0] += self.scales[il] * val * _pdf
    
    def value_shape(self):
        return ()

class PiecewiseLinearOverDatapoints(df.UserExpression):
    """
    A class for defining FEniCS expression that is piece-wise linear over a given data-points: ts, vals.
    IMPORTANT:
        The expression gets updated by either calling "self.(t)" or by setting self.t=t .
    """
    def __init__(self, ts, vals, _degree=0, t0=0., _tol=1e-8 \
                 , _plot=True, _res=None, lab_x='t', lab_y='u', sz=14, _tit='Piecewise linear expression', marker='o' , linestyle='--' \
                    , _save=True, _path='./', _name='Piecewise_linear_expression', _format='.png', _show_plot=True):
        assert len(ts)==len(vals)
        super().__init__(degree=_degree)
        self.ts = ts
        self.vals = vals
        self._tol = _tol
        self.t0 = t0 # initial time
        self.t = t0
        if _plot:
            # _res = len(ts) - 1 if _res is None else _res
            plot_expression_of_t(self, t0=self.t0, t_end=self.ts[-1], ts=self.ts, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit\
                                 , _save=_save, _path=_path, _name=_name, _format=_format, marker=marker, linestyle=linestyle, _show_plot=_show_plot)
    def __call__(self, t):
        self.t = t
    def _value_at_t(self, t):
        if t < self.ts[0] + self._tol: # before first entry of self.ts
            if abs(self.ts[0])<self._tol: # almost zero
                v = self.vals[0]
            else:
                v = t * self.vals[0] / self.ts[0]
        else:
            for i in range(1,len(self.ts)): # refers to end of time-intervals
                t1 = self.ts[i-1]; t2 = self.ts[i]
                if t>t1-self._tol and t<t2+self._tol:
                    a = (t-t1)/(t2-t1); b = 1.-a
                    v = a * self.vals[i] + b * self.vals[i-1]
                    break
        return v
    def eval(self, values, x):
        values[0] = self._value_at_t(self.t)
    def value_shape(self):
        return ()

class SpatialExpressionFromPython(df.UserExpression):
    """
    Given a python callable 'x_callable' that evaluates a desired expression at any spatial points 'x',
    this class builds an equivalent dolfin Expression 'expr', which can be evaluated as usual (expr(x)).
    """
    def __init__(self, x_callable, dim, _degree=0, **kwargs):
        assert isinstance(dim, int) and dim>=0
        self.dim = dim # must be set first, since it is used in 'value_shape'.
        super().__init__(degree=_degree, **kwargs)
        assert callable(x_callable)
        self.x_callable = x_callable
    def eval(self, values, x):
        if self.dim==0: # scalar expression
            values[0] = self.x_callable(x)
        else: # vector expression
            values[:] = self.x_callable(x)
    def value_shape(self):
        return () if self.dim==0 else (self.dim, )

class TimeVaryingExpressionFromPython(df.UserExpression):
    def __init__(self, t_callable, t0=0., T=1.0, _degree=1, nested_expr=df.Expression('val', val=1.0, t=0.0, degree=0) \
                 , _plot=True, _res=1000, lab_x='t', lab_y='u', sz=14, _tit='Plot over time', marker='', linestyle='-'  \
                    , _save=True, _path='./', _name='plot', _format='.png', _show_plot=True \
                     , **kwargs):
        super().__init__(degree=_degree, **kwargs)
        assert callable(t_callable)
        self.t_callable = t_callable # a callable of t
        self.t0 = t0 # initial time
        self.Tend = T # end-time
        self.t = t0 # time
        self.nested_expr = nested_expr
        if _plot:
            plot_expression_of_t(self, t0=self.t0, t_end=self.Tend, _res=_res, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit \
                                 , _save=_save, _path=_path, _name=_name, _format=_format, marker=marker, linestyle=linestyle, _show_plot=_show_plot)
    def eval(self, values, x):
        self.nested_expr.t = self.t
        values[0] = self.t_callable(self.t) * self.nested_expr(x)
    def value_shape(self):
        return ()

class MeshMaskExpression(df.UserExpression):
    """
    Given a mesh, the value of this fenics expression at any point is either of:
        1 , if the point is inside the mesh (including boundaries),
        0 , otherwise.
    For meshes with complicated boundaries, a higher interpolation degree (>=2) is recommended.
    """
    def __init__(self, mesh, degree=2, **kwargs):
        super().__init__(degree=degree, **kwargs)
        self._mesh = mesh
        self._tree = self._mesh.bounding_box_tree()
        self._num_cells = self._mesh.num_cells()
    def eval(self, value, x):
        aa = self._tree.compute_first_entity_collision(df.Point(x))
        if 0 <= aa < self._num_cells:
            value[0] = 1
        else:
            value[0] = 0
    def value_shape(self):
        return ()

def plot_expression_of_t(expr, t0, t_end, ts=None, _res=1000, lab_x='t', lab_y='u', sz=14, _tit='Plot'\
                    , _save=True, _path='./', _name='plot', _format='.png', marker='' , linestyle='-', _show_plot=True):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid") # style must be one of: white, dark, whitegrid, darkgrid, ticks
    except ModuleNotFoundError:
        print(f"\n\n\t{'-' * 70}\n\tWARNING: It is recommended to install 'seaborn' to get nicer plots.\n\t{'-' * 70}\n\n")
    ts, us = evaluate_expression_of_t(expr, t0=t0, t_end=t_end, ts=ts, _res=_res)
    plt.figure()
    plt.plot(ts, us, marker=marker, linestyle=linestyle)
    plt.title(_tit, fontsize=sz)
    plt.xlabel(lab_x, fontsize=sz)
    plt.ylabel(lab_y, fontsize=sz)
    plt.xticks(fontsize=sz)
    plt.yticks(fontsize=sz)
    if _save:
        import os
        if not os.path.exists(_path):
            os.makedirs(_path)
        plt.savefig(_path + _name + _format, bbox_inches='tight', dpi=300)
    if _show_plot:
        plt.ion()
        plt.show(block=False)
    else:
        plt.ioff()
    
def evaluate_expression_of_t(expr, t0=0.0, t_end=1.0, ts=None, _res=1000):
    if ts is None:
        ts = np.linspace(t0, t_end, _res + 1)
    mesh = df.UnitIntervalMesh(1)
    e = df.FiniteElement('R', mesh.ufl_cell(), degree=0)
    V = df.FunctionSpace(mesh, e)
    v = df.TestFunction(V)
    ff = expr * v * df.dx
    us = []
    for tt in ts:
        expr.t = tt
        us.append(sum(df.assemble(ff).get_local()))
    return ts, us

def ramp_expression(f_max, t0=0.0, T=1.0, _degree=1 \
                    , _plot=True, _res=1000, lab_x='t', lab_y='u', sz=14, _tit='Ramp' \
                    , _save=True, _path='./', _name='ramp', _format='.png', _show_plot=True):
    """
        f_max: maximum value of the function
    """
    expr = df.Expression('f_max * t / T', degree=_degree, t=t0, T=T, f_max=f_max)
    if _plot:
        plot_expression_of_t(expr, t0=t0, t_end=T, _res=_res, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit\
                                 , _save=_save, _path=_path, _name=_name, _format=_format, _show_plot=_show_plot)
    return expr

def cyclic_expression(f_max, N=1, t0=0.0, T=1.0, _degree=1, _type='sin' \
                    , _plot=True, lab_x='t', lab_y='u', sz=14, _tit='Cyclic' \
                    , _save=True, _path='./', _name='cyclic', _format='.png', _show_plot=True):
    """
    f_max: maximum value of the function
    N: the number of cycles
    _type:
        "sin": sinusoidal
        "cos": cosinusoidal
        "zigzag". zigzag
    """
    _res = int(N * 1000) + 1
    assert(type(N)==int or type(N)==float)
    if _type == 'sin':
        expr = df.Expression('f_max * sin(N * 2 * p * t / T)', degree=_degree, t=t0, T=T, f_max=f_max, p=np.pi, N=N)
    elif _type == 'cos':
        expr = df.Expression('f_max * cos(N * 2 * p * t / T)', degree=_degree, t=t0, T=T, f_max=f_max, p=np.pi, N=N)
    elif _type == 'zigzag':
        expr = df.Expression('(cos(N * 2 * p * t / T) <= 0.0) ? -r*f_max*(t-T*floor((ceil(r*t))/2)/(2*N)) : r*f_max*(t-T*floor((ceil(r*t))/2)/(2*N))', degree=_degree, t=t0, T=T, f_max=f_max, p=np.pi, N=N, r=4*N/T)            
    if _plot:
        plot_expression_of_t(expr, t0=t0, t_end=T, _res=_res, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit\
                                 , _save=_save, _path=_path, _name=_name, _format=_format, _show_plot=_show_plot)
    return expr

    
def scalar_switcher_expression(intervals_bounds, switch_vals, _degree=0, nested_expr=df.Expression('val', val=1.0, t=0.0, degree=0)\
                               , _plot=True, _res=1000, lab_x='t', lab_y='u', sz=14, _tit='Plot' \
                                   , _save=True, _path='./', _name='plot', _format='.png', _show_plot=True):
    assert len(switch_vals) == len(intervals_bounds) - 1
    def switcher(t):
        value="not_yet_evaluated"
        if t>=intervals_bounds[0] and t<=intervals_bounds[1]:
            value = switch_vals[0]
        i = 1
        while value=="not_yet_evaluated" and i<len(switch_vals):
            if t>intervals_bounds[i] and t<=intervals_bounds[i+1]:
                value = switch_vals[i]
            i += 1
        if value=="not_yet_evaluated":
            raise RuntimeError('The scalar switcher cannot be evaluated at a point outside of the provided intervals.')
        return value
    expr = TimeVaryingExpressionFromPython(t_callable=switcher, t0=intervals_bounds[0], T=intervals_bounds[-1], _degree=_degree, nested_expr=nested_expr \
                 , _plot=_plot, _res=_res, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit \
                    , _save=_save, _path=_path, _name=_name, _format=_format, _show_plot=_show_plot)
    return expr