import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid") # style must be one of: white, dark, whitegrid, darkgrid, ticks
except ModuleNotFoundError:
    print(f"\n\n\t{'-' * 70}\n\tWARNING: It is recommended to install 'seaborn' to get nicer plots.\n\t{'-' * 70}\n\n")

from pathlib import Path
ROOT = Path(__file__).parent.parent # The root directory of the whole repository

class CollectPaths:
    def __init__(self, _name):
        self.ROOT = ROOT
        self._name = _name
        aa = list(self.ROOT.glob("**/" + self._name))
        self.pths = [a.resolve() for a in aa]
    def add_script(self, script_collector):
        for p in script_collector.pths:
            if p not in self.pths:
                self.pths.append(p)
    def add_path(self, pth):
        if pth not in self.pths:
            self.pths.append(pth)
    @staticmethod
    
    def merge_pths(list_of_pths):
        assert isinstance(list_of_pths, list)
        assert all([isinstance(p, list) for p in list_of_pths])
        pths = []
        for pths in list_of_pths:
            for p in pths:
                if p not in pths:
                    pths.append(p)
        return pths

    @staticmethod
    def merge_pths_of_collectors(collectors):
        assert isinstance(collectors, list)
        assert all([isinstance(c, CollectPaths) for c in collectors])
        pths = []
        for c in collectors:
            for p in c.pths:
                if p not in pths:
                    pths.append(p)
        return pths

pth_general = CollectPaths('general.py')

def rotate_points_around_point_2D(XYs, theta, origin=np.array([[0, 0]])):
    """
    It rotates a bunch of points (XYs) around a given point (origin) for theta 'radians'.
    """
    XYs = np.array(XYs)
    assert XYs.shape[1] == 2
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, s], [-s, c]])
    return np.dot(XYs - origin, rot_matrix) + origin

def determinant_or_pseudo_determinant(A, return_log=False, zero_eigen_value_thr=1e-10):
    minus_ps = [100, 200, 300, 400, 500]
    p_max = max([p for p in minus_ps if not np.log(10.**(-p))==-np.inf])
    zero_det_thr = 10. ** (- p_max)
    det = np.linalg.det(A)
    if det > zero_det_thr:
        if return_log:
            return np.log(det)
        else:
            return det
    else: # we compute pseudo-determinant
        print("\n\tWARNING:\n\t\tThe matrix given is very close to singular, therefore, a pseudo-determinant is computed.")
        evs = np.linalg.eigvals(A)
        evs.sort()
        if return_log:
            log_det = 0.
            for ev in evs:
                if ev > zero_eigen_value_thr:
                    log_det += np.log(ev)
            if np.imag(log_det) > zero_eigen_value_thr:
                print("\n\tWARNING:\n\t\tThe pseudo-determinant of the matrix given has turned out to be complex (log_det={log_det}). Only the real part is returned.")
            return np.real(log_det)
        else:
            det = 1.
            for ev in evs:
                if ev > zero_eigen_value_thr:
                    det *= ev
            if np.imag(det) > zero_eigen_value_thr:
                print("\n\tWARNING:\n\t\tThe pseudo-determinant of the matrix given has turned out to be complex ({det}). Only the real part is returned.")
            return np.real(det)

def static_condensation(A, sub_dofs):
    """
    https://en.wikipedia.org/wiki/Guyan_reduction
    https://en.wikipedia.org/wiki/Schur_complement
    """
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert max(sub_dofs) < A.shape[0]
    sub_dofs2 = [i for i in range(A.shape[0]) if i not in sub_dofs]
    Ar = A[np.ix_(sub_dofs, sub_dofs)]
    B = A[np.ix_(sub_dofs, sub_dofs2)]
    C = A[np.ix_(sub_dofs2, sub_dofs)]
    D = A[np.ix_(sub_dofs2, sub_dofs2)]
    return Ar - B @ np.linalg.inv(D) @ C

class MemoryTracker:
    def __init__(self):
        self.memory = []
        self.remarks = []
        self.pid = os.getpid()
        self("init")
    def __call__(self, remark):
        import psutil
        py = psutil.Process(self.pid)
        gb = py.memory_info()[0] / 2.0 ** 30
        self.remarks.append(remark)
        self.memory.append(gb)
    def plot_summary(self):
        m = np.asarray(self.memory)
        plt.plot(self.remarks, m - m[0], "-kx")
        plt.bar(self.remarks[1:], np.diff(m))
        plt.xticks(rotation=90)
        plt.ylabel("memory [GB]")
        plt.show()
    def __str__(self):
        s = ""
        for remark, mem in zip(self.remarks, self.memory):
            s += f"{remark:20}{mem-self.memory[0]:5.3f} GB\n"
        return s

def exp_spatial_correlation(points, cor_length=None, cor_length_scale=0.):
    """
    If the cor_length is NOT given (is None), it is set to:
        'cor_length_scale * Maximum of dual distances between all points',
        so, 'cor_length_scale' is between 0 and 1 .
    """
    import scipy.spatial as scispa
    distances = scispa.distance.cdist(points, points, metric='euclidean')
    if cor_length is None:
        assert (0. < cor_length_scale <= 1.0)
        cor_length = np.max(distances) * cor_length_scale
    return cor_length, np.exp(- distances / cor_length)

def compute_jacobian_cd(f, X, X_ids=None, _eps=1e-5, zero_thr=1e-10):
    """
    f: A callable with:
        input:  A long vector (X)
        output: Either a long vector or a dictionary whose values are long vectors.
    X: A long vector as the input to f (at which the jacobian is computed).
    X_ids: IDs of X with respect to which the jacobian is computed.
    """
    if X_ids is None:
        X_ids = range(len(X))
    l = len(X_ids)
    jac_cd = None # will be allocated as soon as 'f' is called first.
    def _allocate_jac_cd(an_f_output):
        if isinstance(an_f_output, dict):
            jac_cd = {}
            for k, f in an_f_output.items():
                jac_cd[k] = np.zeros((f.size, l))
        else: # array
            jac_cd = np.zeros((an_f_output.size, l))
        return jac_cd
    print(f"\n----- Computation of jacobian using central difference started with respect to {l} input entries. -----")
    for i_cd in range(l):
        ix = X_ids[i_cd]
        xi = X[ix]
        x_minus = copy.deepcopy(X)
        x_plus = copy.deepcopy(X)
        if abs(xi) <= zero_thr: # too close to zero
            dxi = 2. * _eps
            x_minus[ix] = xi - dxi/2.
            x_plus[ix] = xi + dxi/2.
        else:
            dxi = 2. * _eps * abs(xi)
            x_minus[ix] = xi - dxi/2.
            x_plus[ix] = xi + dxi/2.
        f_minus = copy.deepcopy(f(x_minus))
        f_plus = copy.deepcopy(f(x_plus))
        if jac_cd is None:
            jac_cd = _allocate_jac_cd(f_minus)
        if isinstance(f_plus, dict):
            for k in f_plus.keys():
                jac_cd[k][:, i_cd] = (f_plus[k] - f_minus[k]) / dxi
        else:
            jac_cd[:, i_cd] = ((f_plus - f_minus) / dxi)
    print(f"\n----- Computation of jacobian using central difference ended. -----")
    return jac_cd

def linear_interpolation_with_nearest_extrapolation(points, values, new_points):
    """
    This is NOT suitable if the domain (union of the given points) is non-convex:
        REASON: Some of new_points can be out of domain (which means they must be treated by 'extrapolation'),
        but they are still treated by 'interpolation', since the domain is non-convex.
    """
    from scipy.interpolate import griddata
    vals_nearest = griddata(points, values, new_points, method='nearest') # it somehow does extrapolation, too.
    fv = int(100 * np.max(abs(values))) # some filled value that is certainly not in data.
    vals = griddata(points, values, new_points, method='linear', fill_value=fv)
    _ids = np.where(vals==fv)[0] # detect extrapolation positions.
    vals[_ids] = vals_nearest[_ids] # replace 'fv' with nearest value at extrapolation positions detected.
    return vals

def nearest_values(points, values, new_points):
    from scipy.interpolate import griddata
    vals_nearest = griddata(points, values, new_points, method='nearest') # it somehow does extrapolation, too.
    return vals_nearest

def interANDextrapolation_with_Rbf(points, values, new_points, function='multiquadric'):
    assert points.shape[1]==new_points.shape[1]
    assert points.shape[0]==values.shape[0]
    assert points.shape[1] <= 3
    from scipy.interpolate import Rbf
    _mode = '1-D' if len(values.shape)==1 else 'N-D'
    _d = 3 - points.shape[1] # missing dimensions (up to 3)
    zz = np.zeros((points.shape[0], 1))
    zz_new = np.zeros((new_points.shape[0], 1))
    for _i in range(_d):
        points = np.concatenate((points, zz), axis=1)
        new_points = np.concatenate((new_points, zz_new), axis=1)
    rbfi = Rbf(points[:,0], points[:,1], points[:,2], values, function=function, mode=_mode)
    return rbfi(new_points[:,0], new_points[:,1], new_points[:,2])

def interANDextrapolation_with_weightedLS(points, values, new_points, function='linear' \
                                          , points_tol=None, _print=True):
    """
    Two cases:
        new_points coinciding the given points:
            --> returns the same value
        new_points other than given points:
            --> 'FOR EACH new_point' returns evaluation of an ansatz function (of the given type; i.e. linear by default)
                that will be fitted using weighted-least-square with the weight function being
                inverse of norm-2 distances of given points to each certain new_point.
    """
    assert points.shape[1]==new_points.shape[1]
    assert points.shape[0]==values.shape[0]
    values = values.reshape(values.shape[0], -1)
    from scipy.optimize import curve_fit
    if points_tol is None:
        points_tol = (np.max(points) - np.min(points)) / 10000.
    def _a_weight_vector(p0, points, _power=2., _factor=1.):
        """
        It returns a vector with size of the number of points, each entry being a weight associated to that point.
        This weight vector is going to become sigma^(-2), where sigma is uncertainty of data (value) at any point.
        The more far away a point is w.r.t. p0, its weight is smaller; i.e. the uncertainty is larger.
        """
        weights = [_factor * (np.linalg.norm(p - p0) ** -_power) for p in points]
        return np.array(weights)
    if function=='linear':
        def _ansatz_function(points, *args):
            """
            Returns evaluation of the linear function:
                f = args[0] * point[0] + args[1] * point[1] + ... + args[-1]
            for all given points.
            """
            assert len(points[0]) + 1 == len(args)
            vals = []
            for point in points:
                val = args[-1]
                for i in range(len(args) - 1):
                    val += args[i] * point[i]
                vals.append(val)
            return vals
        p0 = (len(points[0]) + 1) * (1.,)
    else:
        raise NotImplementedError(f"The given type of ansatz function '{function}' is not implemented.")
    
    vals = []
    lv = values.shape[1]
    for pn in new_points:
        _found = False
        for ii, pp in enumerate(points):
            if np.linalg.norm(pn - pp) < points_tol:
                _found = True
                vals.append(values[ii])
                break
        if not _found:
            ws = _a_weight_vector(pn, points)
            sigma = ws ** (-2)
            val = []
            for jj in range(lv):
                _val = values[:, jj]
                popt, pcov = curve_fit(_ansatz_function, points, _val, p0, sigma=sigma, absolute_sigma=True)
                u_ = _ansatz_function([pn], *popt)[0]
                val.append(u_)
            vals.append(val)
            if _print:
                print(f"The data-point {pn} is extrapolated from the data.")
    return np.array(vals)

class SumRowsIn2DArray:
    def __init__(self, A, rows_groups, first_fix_row=None):
        assert (type(A)==np.ndarray)
        sh = A.shape
        assert (len(sh)==2)
        for g in rows_groups:
            for r in g:
                assert (r < sh[0])
        
        if first_fix_row is None:
            first_fix_row = sh[0]
        self.first_fix_row = first_fix_row
        
        joint_rows = [r for g in rows_groups for r in g]
        single_rows = [[i] for i in range(0, self.first_fix_row) if i not in joint_rows]
        fix_rows = [i for i in range(self.first_fix_row, sh[0]) ]
        
        self.rows_groups = rows_groups + single_rows
        self.A = A
        self._A_summed = np.zeros((len(self.rows_groups) + len(fix_rows), sh[1]))
        for i, r in enumerate(fix_rows):
            self._A_summed[len(self.rows_groups) + i, :] = self.A[r, :]
    @property
    def summed(self):
        for i, g in enumerate(self.rows_groups):
            self._A_summed[i,:] = 0.0
            for r in g:
                self._A_summed[i,:] += self.A[r]
        return self._A_summed

class Split2DArray:
    def __init__(self, A, rows_groups):
        assert (type(A)==np.ndarray)
        sh = A.shape
        assert (len(sh)==2)
        if rows_groups==[[]]:
            rows_groups=[]
        for g in rows_groups:
            for r in g:
                assert (r < sh[0])
        self.A = A
        self.rows_groups = rows_groups
        
        joint_rows = [r for g in rows_groups for r in g]
        self.rows_rest = [[i] for i in range(0, sh[0]) if i not in joint_rows] # single rows
        
        self._A_groups = np.zeros((len(self.rows_groups), sh[1]))
        self._A_rest = np.zeros((len(self.rows_rest), sh[1]))
        
    def split(self):
        for i, g in enumerate(self.rows_groups):
            self._A_groups[i,:] = 0.0
            for r in g:
                self._A_groups[i,:] += self.A[r]
        for i, r in enumerate(self.rows_rest):
            self._A_rest[i,:] = self.A[r] # single row
        return self._A_groups, self._A_rest

class ConcatenateBlockArray2D:
    """
    The instance of this class has a property "assembled" which will be concatenated from the actual value of a block array (A).
    self.A:
        the intended block array, whose all entries are:
            - either None
            - or 2-D np.array objects of the SAME shape
    self.assembled: 
        the assembled FULL 2-D array concatenated from blocks of self.A
    """
    def __init__(self, A):
        self.A = A
        self.Sh = self.A.shape
        self.sh = self.A[0][0].shape
        self._A = np.zeros((self.Sh[0]*self.sh[0], self.Sh[1]*self.sh[1])) # initiate
    @property
    def assembled(self):
        for i in range(self.Sh[0]):
            for j in range(self.Sh[1]):
                if self.A[i][j] is not None:
                    self._A[i*self.sh[0]:(i+1)*self.sh[0], j*self.sh[1]:(j+1)*self.sh[1]] = self.A[i][j][:,:]
        return self._A

class CallToGetUpdatedAttributes:
    """
    This is a helper class, whose instance will be called (without any argument) in order to return the last updated attributes of an object.
    This is needed because the attributes might have changed however any previous copy/reference of them have NOT been updated accordingly.
        obj: an object (instance) whose updated attributes are intended to be returned.
        attributes_list: list of strings denoting the attributes of obj (of course, must match the class from which obj is instanced).
    The __call__ method returns:
        a list containing the last updated values of the desired attributes. In case of only one attribute, it is returned directly (not in a list).
    """
    def __init__(self, obj, attributes_list):
        self.obj = obj
        self.attrs_list = attributes_list
    def __call__(self):
        atts = []
        for a in self.attrs_list:
            atts.append(getattr(self.obj, a))
        if len(self.attrs_list)==1:
            return atts[0]
        else:
            return atts

def animate_list_of_2d_curves(x_fix, ys_list, _delay=200 \
                              , xl='x', yl='y', _tit='Animation', _marker='', _linestyle='solid' \
                              , _path='./', _name='animate', _format='.gif', dpi=200):
    fig = plt.figure()
    dx = max(x_fix) - min(x_fix)
    dy = np.max(ys_list) - np.min(ys_list)
    dx_ = 0.02 * dx
    dy_ = 0.02 * dy
    ax = plt.axes(xlim=(min(x_fix) - dx_, max(x_fix) + dx_), ylim=(np.min(ys_list) - dy_, np.max(ys_list) + dy_) )
    ax.set_title(_tit)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    line, = ax.plot([], [], marker=_marker, linestyle=_linestyle)
    def init_():
        line.set_data([], [])
        return line,
    def animate_(i):
        line.set_data(x_fix, ys_list[i])
        return line,
    anim = FuncAnimation(fig, animate_, init_func=init_,
                        frames=len(ys_list), interval=_delay, blit=True)
    anim.save(_path+_name+_format, writer='imagemagick', dpi=dpi)

def plots_together(xs, ys, labels=None, x_label='x', y_label='y', _tit='PLOTS', _name='plots', _path='./', _format='.png', sz=14):
    assert len(xs)==len(ys)
    if labels is None:
        labels = [str(i+1) for i in range(len(xs))]
    assert len(xs)==len(labels)
    plt.figure()
    _markers = int(np.ceil(len(xs)/23)) * [",", ".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    for i,x in enumerate(xs):
        plt.plot(x, ys[i], marker=_markers[i], label=labels[i])
    plt.title(_tit, fontsize=sz)
    plt.xlabel(x_label, fontsize=sz)
    plt.ylabel(y_label, fontsize=sz)
    plt.xticks(fontsize=sz)
    plt.yticks(fontsize=sz)
    plt.legend()
    plt.savefig(_path + _name + _format)
    plt.show()

def write_attributes(obj, _name='object', _path='./', _format='.txt'):
    make_path(_path)
    with open(_path + 'Attributes_of_' + _name + _format, 'w') as f:
        f.write('Attributes of ' + _name + ':\n')
        f.write('\n'.join(["%s = %s" % (k,v) for k,v in obj.__dict__.items()]))

class LoggerSetup:
    def __init__(self, file_name='loger_unnamed.log'):
        self.file_name = file_name
        
        # console logger (log to console)
        self.console = logging.getLogger('console')
        self.console.setLevel(logging.DEBUG)
        # create and add console handler
        self.console.handlers = []
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        self.console.addHandler(ch)
        
        # file logger (log to file)
        self.file = logging.getLogger('file')
        self.file.setLevel(logging.DEBUG)
        self.file.handlers = []
        # create and add file handler
        fh = logging.FileHandler(self.file_name, mode='w+') # "w+" is for overwriting
        fh.setLevel(logging.DEBUG)
        self.file.addHandler(fh)

def whoami(): # return the name of current method
    import sys
    return sys._getframe(1).f_code.co_name
    
def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_directory(directory):
    """
    Clean/remove all files located within a directory.
    """
    for file_name in os.listdir(directory):
        file = directory + file_name
        if os.path.isfile(file):
            os.remove(file)

def find_mixup(XY, XY_mixed, tol):
    """
    XY and XY_mixed both are list of np.array entries.
    They both have the same entries, but sorted differently.
    The method returns "ids", such that:
        XY_mixed[i] = XY[ids[i]] for all possible "i"s
    IMPORTANT assumption:
        Either of XY and XY_mixed has NO duplicate entries (in each of them, all entries are different).
    """
    assert(len(XY) == len(XY_mixed))
    ids = find_among_points(points=XY_mixed, points0=XY, tol=tol)
    return ids

def almost_equal_arrays(A1, A2, tol, _method='norm2', raise_error=True):
    """
    A1 is the reference array
    A2 is the array to be compared with A1
    It returns:
        B: boolean of whether A1 is almost equal to A2
        max_err: the maximum error between A1 and A2
    """
    assert A1.shape==A2.shape
    B = True
    max_err = 0.0
    
    if _method == 'norm2':
        if np.linalg.norm(A1)<=1e-8 or np.linalg.norm(A2)<=1e-8:
            reff = 1.0
        else:
            reff = np.linalg.norm(A1)
        err = np.linalg.norm(A1-A2) / reff
        max_err = err
        if not err < tol:
            B = False
            if raise_error:
                raise AssertionError("The given arrays are not identical with relative norm-2 error=" + str(err) + " .")
    elif _method == 'entry':
        sizes = A1.shape
        if len(sizes)==1:
            print(' --------------- to be developed for 1D ---- ')
        elif len(sizes)==2:
            for i in range(sizes[0]):
                for j in range(sizes[1]):
                    if A1[i][j]<1e-9 or A2[i][j]<1e-9:
                        reff = 1.0
                    else:
                        reff = abs(A1[i][j])
                    err = abs(A1[i][j] - A2[i][j]) / reff
                    max_err = max(max_err, err)
                    if not err < tol:
                        B = False
                        if raise_error:
                            raise AssertionError("The given arrays are not identical with relative error=" + str(err) + "at entry of i,j=" + str(i) + "," + str(j) + " .")
        else:
            print(' --------------- to be developed for higher dimensions ---- ')
    
    return B, max_err
        
def overlay_2plots(back, front, save_to):
    try:
        from PIL import Image
    except ImportError:
        import Image
    background = Image.open(back)
    overlay = Image.open(front)
    imbg_width, imbg_height = background.size
    overlay = overlay.resize((imbg_width, imbg_height), Image.LANCZOS)
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(save_to, "PNG")        
    
def add_unique_points(new_points, points0=[], _eps=1e-12):
    """
    Add new_points to points0 if they are minimum "_eps" far away from themselves and points0. --> avoid repeated points.
    """
    if not isinstance(points0, list):
        points0 = points0.tolist()
    added_points = []
    for p in new_points:
        _new = True
        for p0 in points0:
            if np.linalg.norm(p0 - p) < _eps:
                _new=False
                break
        if _new:
            points0.append(p)
            added_points.append(p)
    return points0, added_points

def find_among_points(points, points0, tol):
    """
    points: the target points to be found.
    points0: the points among which we want to find target points.
    tol: a tolerance which decides two points are matching.
    Returns:
        _ids: the IDs of entries in points0 which correspond to target points (with the same order as points).
            If any point (of points) is NOT found in points0, then the corresponding _ids will be set to None.
    """
    if len(points0)==0:
        _ids = [None for i in range(len(points))]
    else:
        _ids = []
        for p in points:
            _id = np.linalg.norm(points0 - p, axis = 1).argmin()
            if np.linalg.norm(points0[_id] - p) > tol:
                _id = None
            _ids.append(_id)
    return _ids

def contour_plot_2d_irregular(xs, ys, zs, res=0.5, _tits=['Title'], xl='', yl='', levels=15 \
                                  , _path='./', _name='contourplot', _format='.pdf', dpi=100):
    """
    zs: A list of several z (ideally for 1 or 2 or 3 z entries. A more number of entries needs adjustment of subplots).
    x, y, zs: 1-D arrays of the same length representing some arbitrary irregular data points.
    res: The resolution (number of grid per length) at every direction.
    
    Source: https://matplotlib.org/stable/gallery/images_contours_and_fields/irregulardatagrid.html
    """
    fig = plt.figure()
    
    nz = len(zs)
    _min = 1e20
    _max = -1e20
    for i,z in enumerate(zs):
        _min=min(_min, min(z))
        _max=max(_max, max(z))
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    cmap = cm.get_cmap('RdBu_r', lut=levels)
    normalizer = Normalize(_min,_max)
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    
    axs = []
    for i,z in enumerate(zs):
        x = xs[i]; y = ys[i];
        ax = fig.add_subplot(nz,1,i+1)
        plt.subplots_adjust(hspace=nz*0.5)
        ax.axes.set_aspect('equal')
        
        # plot scatter of data itself
        ax.plot(x, y, 'ko', ms=2)
        
        # Create grid values
        x0 = min(x); x1 = max(x)
        y0 = min(y); y1 = max(y)
        ngridx = int(round(res * (x1-x0)))
        ngridy = int(round(res * (y1-y0)))
        xi = np.linspace(x0, x1, )
        yi = np.linspace(y0, y1, ngridy)
        
        # WAY 1: Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        import matplotlib.tri as tri
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
            # The following would be an alternative to the four lines above:
            # from scipy.interpolate import griddata
            # zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        ax.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
        cntr = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, norm=normalizer) # for colours
        
        # WAY 2: Directly supply the unordered, irregularly spaced coordinates to tricontour.
        # ax.tricontour(x, y, z, levels=levels, linewidths=0.5, colors='k')
        # cntr = ax.tricontourf(x, y, z, levels=levels, cmap=cmap, norm=normalizer) # for colours
        
        ax.set(xlim=(x0, x1), ylim=(y0, y1))
        ax.set_title(_tits[i])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        axs.append(ax)
    
    cbar = fig.colorbar(im, ax=axs, orientation="vertical")
    # cbar = fig.colorbar(im, ax=axs, orientation="horizontal")
    # cbar.set_ticks(np.arange(...))
    # cbar.set_ticklabels(['low', 'medium', 'high', ...])
    
    plt.savefig(_path + _name + _format, bbox_inches='tight', dpi=dpi)
    plt.show()
    
def plot_interpolated_objective(callable_objective, X0, X1, res=20 \
                                , xl='Interpolation factor', yl='Objective', tit='Interpolated objective' \
                                   , _save=True, _path='', _name='interpolated_objective', _format='.png', dpi=200):
    dX = np.array(X1) - np.array(X0)
    ws = np.linspace(0.0, 1.0, res+1)
    os = []
    for w in ws:
        X = X0 + w * dX
        os.append(callable_objective(X))
    plt.figure()
    plt.plot(ws, os, marker='*', linestyle='')
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(tit)
    if _save:
        plt.savefig(_path + _name + _format, bbox_inches='tight', dpi=dpi)
    plt.show()

def repeat_periodic_ids(p_ids, N):
    """
    Every entry of the list "p_ids" is a list of "ids".
    The merge of all lists in "p_ids" must include EVERY index between 0 and some number ONLY once. 
    N is the number of the full "ids" generated.
    Length of the merged "ids" must be a counter of N.
    
    example:
        given p = [[3,1], [4,2,5,0]]
        , we get:
            repeat_periodic_ids(p, 18) =
            [[3, 1, 9, 7, 15, 13], [4, 2, 5, 0, 10, 8, 11, 6, 16, 14, 17, 12]]
    """
    n_ps = len(p_ids) # number of different sets of periodic ids
    m_ids = [] # merged ids
    for p in p_ids:
        m_ids += p
    n_ids = len(m_ids)
    
    assert(N%n_ids==0)
    assert(max(m_ids) + 1 == n_ids)
    assert(sorted(m_ids)==list(range(max(m_ids)+1)))
    
    N = int(N/n_ids) # number of repeatations
    ids = n_ps * [[]]
    for n in range(N):
        shift = n * n_ids
        for j, ids_j in enumerate(p_ids):
            new_ids_j = [i+shift for i in ids_j]
            ids[j] = ids[j] + new_ids_j
    return ids

def area_under_XY_data(Xs, Ys):
    Xs_sorted = []; Ys_sorted = []
    for x, y in sorted(zip(Xs, Ys)):
        Xs_sorted.append(x)
        Ys_sorted.append(y)
    _sum = 0.
    for i in range(1, len(Xs_sorted)):
        delta_x = Xs_sorted[i] - Xs_sorted[i - 1]
        delta_y_ave =  (Ys_sorted[i] + Ys_sorted[i-1]) / 2.
        _sum += delta_x * delta_y_ave
    return _sum

def plot_convergence_of_Us(Uss, ch_IDs=None, _path='./'):
    """
    Uss:
        List of arrays each being:
            - for a mesh refinement level (increasing)
            - displacements at any certain number of checkpoints
    """
    assert all(len(us.shape)==3 for us in Uss)
    if ch_IDs is None:
        uss = Uss; ll = 'all'; LL = '(all load-steps)'
    else:
        assert len(ch_IDs)==2
        a1, a2 = ch_IDs
        uss = [us[a1:a2,:,:] for us in Uss]
        ll = f"ls_{a1}_{a2}"; LL = f"(load-steps=[{a1},{a2}])"
    n_meshes = len(uss)
    res_levels = range(1, n_meshes)
    u0_norm = np.linalg.norm(uss[0])
    errs_a = [np.linalg.norm(uss[i+1]-uss[i]) for i in range(n_meshes-1)]
    errs_r0 = [ee/u0_norm for ee in errs_a]
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.subplots_adjust(top=0.85, wspace=0.65)
    
    axes[0].plot(res_levels, errs_a, linestyle='--', marker='o')
    axes[0].set_xlabel('i : refinement level')
    axes[0].set_xticks(res_levels)
    axes[0].set_ylabel("Log ( absolute error ) =\nlog ( | U[i] - U[i-1] | )")
    axes[0].set_yscale('log')
    
    axes[1].plot(res_levels, errs_r0, linestyle='--', marker='o')
    axes[1].set_xlabel('i : refinement level')
    axes[1].set_xticks(res_levels)
    axes[1].set_ylabel("Log ( relative error ) =\nlog ( | U[i] - U[i-1] | / | U[0] | )")
    axes[1].set_yscale('log')
    
    fig.suptitle(f"Error between successive solved displacements\n{LL}")
    plt.savefig(_path + f"mesh_errors_{ll}.png", bbox_inches='tight', dpi=300)
    
    plt.show()