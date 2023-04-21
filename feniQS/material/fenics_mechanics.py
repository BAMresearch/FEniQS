import sys
if './' not in sys.path:
    sys.path.append('./')

import numpy as np
import dolfin as df
import ufl

## GLOBALLY DEFINED
_ss_vector = 'Voigt' # with (2) factor only in shear strains
# _ss_vector = 'Mandel' # with sqrt(2) factor in both shear strains and shear stresses

from feniQS.general.general import CollectPaths
pth_fenics_mechanics = CollectPaths('fenics_mechanics.py')

def ss_dim(constraint):
    constraint_switcher = {
                'UNIAXIAL': 1,
                'PLANE_STRESS': 3,
                'PLANE_STRAIN': 4,
                '3D': 6
             }
    _dim = constraint_switcher.get(constraint, "Invalid constraint given. Possible values are: " + str(constraint_switcher.keys()))
    return _dim

def eps_vector(v, constraint, ss_vector='default'):
    _ss_dim = ss_dim(constraint)
    if ss_vector=='default':
        ss_vector = _ss_vector
    if _ss_dim==1:
        return df.grad(v)
    else:
        e = df.sym(df.grad(v))
        if ss_vector.lower() == 'voigt':
            _fact = 2
        elif ss_vector.lower() == 'mandel':
            _fact = 2 ** 0.5
        
        if _ss_dim==3:
            return df.as_vector([e[0, 0], e[1, 1], _fact * e[0, 1]])
        elif _ss_dim==4:
            return df.as_vector([e[0, 0], e[1, 1], 0, _fact * e[0, 1]])
        elif _ss_dim==6:
            return df.as_vector( [ e[0, 0], e[1, 1], e[2, 2] \
                                 , _fact * e[1, 2], _fact * e[0, 2], _fact * e[0, 1] ] )

def sigma_vector(sigma, constraint, ss_vector='default'):
    _ss_dim = ss_dim(constraint)
    if ss_vector=='default':
        ss_vector = _ss_vector
    if _ss_dim==1:
        return sigma
    else:
        if ss_vector.lower() == 'voigt':
            _fact = 1
        elif ss_vector.lower() == 'mandel':
            _fact = 2 ** 0.5
        
        if _ss_dim==3:
            return df.as_vector([sigma[0, 0], sigma[1, 1], _fact * sigma[0, 1]])
        elif _ss_dim==4:
            return df.as_vector([sigma[0, 0], sigma[1, 1], 0., _fact * sigma[0, 1]])
            # IMPORTANT: In plane strain, the out of plane stress must be computed and replaced via: sigma_33 = nu * (sigma_11 + sigma_22) 
        elif _ss_dim==6:
            return df.as_vector( [ sigma[0, 0], sigma[1, 1], sigma[2, 2] \
                                 , _fact * sigma[1, 2], _fact * sigma[0, 2], _fact * sigma[0, 1] ] )

def constitutive_coeffs(E=1000, nu=0.0, constraint='PLANE_STRESS'):
    lamda=(E*nu/(1+nu))/(1-2*nu)
    mu=E/(2*(1+nu))
    if constraint=='PLANE_STRESS':
        lamda = 2*mu*lamda/(lamda+2*mu)
    return mu, lamda

class ModifiedMises:
    """
    Taken from https://git.bam.de/mechanics/ttitsche/damage-models/-/blob/master/fenics_damage/components.py
    """
    
    """ Modified von Mises equivalent strain, see
    de Vree et al., 1995, "Comparison of Nonlocal Approaches in
    Continuum Damage Mechanics"

    Invariants from https://en.wikipedia.org/wiki/Cauchy_stress_tensor

    The members T1, T2, T3 correspond to the term 1,2,3 in the equation
    """
    def __init__(self, nu, k=10.0):
        self.k = k
        self.nu = nu
        self.T1 = (k - 1) / (2. * k * (1. - 2. * nu))
        self.T2 = (k - 1) / (1. - 2. * nu)
        self.T3 = 12. * k / ((1. + nu) * (1. + nu))

    def __call__(self, eps_3d):
        if hasattr(eps_3d, 'ufl_shape'):
            assert (eps_3d.ufl_shape[0] == 3)
            I1 = df.tr(eps_3d)
            J2 = 0.5 * df.tr(df.dot(eps_3d, eps_3d)) - 1. / 6. * I1 * I1
            A = (self.T2 * I1) ** 2 + self.T3 * J2
            A_pos = ufl.operators.Max(A, 1.e-14)
            return self.T1 * I1 + df.sqrt(A_pos) / (2. * self.k)
        elif type(eps_3d)==list:
            assert (len(eps_3d) == 3)
            eps_3d = np.array(eps_3d)            
            I1 = np.trace(eps_3d)
            J2 = 0.5 * np.trace(np.dot(eps_3d, eps_3d)) - 1. / 6. * I1 * I1
            A = (self.T2 * I1) ** 2 + self.T3 * J2
            A_pos = max(A, 1.e-14)
            return self.T1 * I1 + df.sqrt(A_pos) / (2. * self.k)
    
    def d__d_eps(self, eps): # The derivative of "__call__" w.r.t. epsilon, which returns a tensor of the size being equal to the problem dimension.
        ##########################    
        return 1.; ######################### just for pass something (not adjusted yet)

def epsilon(u, _dim=None):
    if _dim==None:
        ns = u.function_space().num_sub_spaces()
        _dim = 1 if ns==0 else ns
    # depending on 1-D or higher-dimensional u, we need to use different formulas since for 1-D neither "sym" nor "Transposition" works in FEniCS.
    if _dim==1:
        return ufl.grad(u)
    else:
        return ufl.sym(ufl.grad(u))

def eps_eq_1d(eps_3d):
    return ufl.algebra.Abs(eps_3d[0, 0])
    # # "ufl.algebra" is used just to not mix with the similar python operator
    
def eps_3d(eps, nu, constraint):
    if hasattr(eps, 'ufl_shape'):
        d = eps.ufl_shape[0]
    elif type(eps)==list or type(eps)==np.ndarray:
        d = len(eps)
    else: # scalar
        d = 1
    
    if d == 1:
        if hasattr(eps, 'ufl_shape'):
            eps = eps[0] # "[0]" is used to convert shape "(1,)" to shape "()" (scalar)
        e_3d = [[eps, 0, 0],
                [0,  0, 0],
                [0,  0, 0]]
    elif d == 2:
        if hasattr(eps, 'ufl_shape'):
            e11 = eps[0, 0]; e12 = eps[0, 1]; e21 = eps[1, 0]; e22 = eps[1, 1];
        else:
            e11 = eps[0][0]; e12 = eps[0][1]; e21 = eps[1][0]; e22 = eps[1][1];
        
        if constraint=='PLANE_STRESS':
            ezz = - nu / (1 - nu) * (e11 + e22)
            e_3d = [[e11, e12, 0],
                          [e21, e22, 0],
                          [0, 0, ezz]]
        elif constraint=='PLANE_STRAIN':
            e_3d = [[e11, e12, 0],
                          [e21, e22, 0],
                          [0, 0, 0]]
    elif d == 3:
        e_3d = eps # here might be as dolfin.tensor already (has ufl_shape)!
    if not hasattr(eps_3d, 'ufl_shape'):
        e_3d = df.as_tensor(e_3d)
    return e_3d # of type "ufl"


