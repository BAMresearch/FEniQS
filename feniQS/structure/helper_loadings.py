import sys
if './' not in sys.path:
    sys.path.append('./')

from feniQS.fenics_helpers.fenics_expressions import *

pth_helper_loadings = CollectPaths('helper_loadings.py')
pth_helper_loadings.add_script(pth_fenics_expressions)

class ParsLoading:
    def __init__(self, level):
        """
        Parameters for specifying a loading curve (of time). Closely connected to the "time_varying_loading" method below.
        case: 'sin' or 'zigzag' or 'ramp' (cyclic ones are symetrical w.r.t. the y=0 axis)
            If case='ramp', a list of "scales" having several entries will lead to a discontinious loading. 
        level: float number for the total level of loading
        scales: scales for getting loading peaks from the level; i.e.: loading_peaks = [level*s for s in scales]
        N: Number of cycles (only relevant if case is NOT "ramp").
            Consider its interaction with "len(self.levels)". If len(scales)>=1, to have a continious curve, N must be: an_integer * 0.5 * len(scales).
        """
        self.case = 'ramp'
        # self.case = 'sin'
        # self.case = 'zigzag'
        
        self.level = level
        # self.scales = np.array([1.0])
        self.scales = np.array([1.0, -1.5, 2.0])
        
        self.N = 1.5
        
        self.T = 1.0
        self.t_end = self.T

def time_varying_loading(fs=1.0, intervals_bounds=None, N=1, t0=0.0, T=1.0, _case='ramp', _degree=1 \
                            , _plot=True, _res=1000, lab_x='t', lab_y='u', sz=14 \
                            , _save=True, _path='./', _name=None, _format='.png'):
    """
    A helper method to return desired typical time-varying loading.
    This uses the module py_fenics.fenics_expressions.
    
    Parameters
    ----------
    fs : TYPE, int / np.array / list
        DESCRIPTION. The peak(s) of loading. Using several different values in "fs" together with _case='ramp' leads to a discontinious loading. The default is 1.0.
    intervals_bounds : TYPE, np.array / list
        DESCRIPTION. A sequence of "n" numbers indicating "n-1" intervals.
                    In case of several values for "fs", this indicates for which interval of time any peak velus in "fs" is considered.
                    The default is None, where the intervals are simply set by deviding the whole time span into len(fs) parts.
    N : TYPE, float / int
        DESCRIPTION. Number of cycles (only relevant if case is NOT "ramp").
        Consider its interaction with len(fs). If len(fs)>=1, to have a continious curve, N must be: an_integer * 0.5 * len(fs). The default is 1.
    t0 : TYPE, float / int
        DESCRIPTION. Initial time. The default is 0.0.
    T : TYPE, float / int
        DESCRIPTION. Final time. The default is 1.0.
    _case : TYPE, string (from a limited number of choices)
        DESCRIPTION. Indicates the loading pattern. The default is 'ramp', with which a list of "fs" having several entries will lead to a discontinious loading. 
    _degree : TYPE, int
        DESCRIPTION. The expression degree. The default is 1.
    
    The rest are for getting a plot and saving it.

    Raises
    ------
    RuntimeError
        DESCRIPTION. When an invalid (undeveloped) loading "_case" is given.

    Returns
    -------
    TYPE, df.Expression
        DESCRIPTION. A dolfin Expression of time(t) which is based on the given pattern, peaks(fs), and number of cycles(N).

    """
    if type(fs)==list or type(fs)==np.ndarray:
        f = 1.0
        _plot0 = False
    else:
        f = fs
        _plot0 = _plot
    def ramp():
        return ramp_expression(f_max=f, t0=t0, T=T, _degree=_degree \
                               , _plot=_plot0, _res=_res, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit \
                                   , _save=_save, _path=_path, _name=_name, _format=_format)
    def sin():
        return cyclic_expression(f_max=f, N=N, t0=t0, T=T, _degree=_degree, _type='sin' \
                               , _plot=_plot0, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit \
                                   , _save=_save, _path=_path, _name=_name, _format=_format)
    def zigzag():
        return cyclic_expression(f_max=f, N=N, t0=t0, T=T, _degree=_degree, _type='zigzag' \
                               , _plot=_plot0, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit \
                                   , _save=_save, _path=_path, _name=_name, _format=_format)
    tit_switcher = {
                'ramp': 'Ramp loading over time',
                'sin': 'Sinusoidal loading over time',
                'zigzag': 'Zigzag loading over time',
             }
    expr_switcher = {
                'ramp': ramp,
                'sin': sin,
                'zigzag': zigzag
             }
    
    _tit = tit_switcher.get(_case, "Invalid case of time-varying loading.")
    if _name is None:
        _name = _tit
    func = expr_switcher.get(_case, "Invalid case of time-varying loading.")
    
    if type(func)!=str:
        expr0 = func()
    else:
        raise RuntimeError(func)
    
    if type(fs)==list or type(fs)==np.ndarray:
        if intervals_bounds is None:
            intervals_bounds = np.linspace(t0, T, len(fs)+1)
        expr = scalar_switcher_expression(intervals_bounds, fs, _degree=_degree, nested_expr=expr0 \
                                          , _plot=_plot, _res=_res, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit \
                                   , _save=_save, _path=_path, _name=_name, _format=_format)
    else:
        expr = expr0
    return expr

if __name__ == "__main__":
    """
    To illustrate how it works.
    """
    _degree = 1
    
    fs = [1,-1.5,2]
    N = 1.5 # if len(fs)>=1, to have a continious curve, N must be: an_integer * 0.5 * len(fs)
    
    e1 = time_varying_loading(fs, N=N, _case='sin')
    e2 = time_varying_loading(fs, N=N, _case='zigzag')    
    e3_1 = time_varying_loading(fs, N=N, _case='ramp') # discontinious if using several "fs" with _case='ramp'
    fs = [1]
    e3_2 = time_varying_loading(fs, N=N, _case='ramp') # Ok with _case='ramp'