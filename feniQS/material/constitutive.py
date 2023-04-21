import sys
if './' not in sys.path:
    sys.path.append('./')

from ufl import grad, nabla_grad, div, nabla_div, sym
from feniQS.material.damage import *
from feniQS.material.fenics_mechanics import *
from feniQS.material.fenics_mechanics import _ss_vector
import numpy as np

"""
    Contain classes/methods regarding:
        - Constitutive relations
"""

pth_constitutive = CollectPaths('constitutive.py')
pth_constitutive.add_script(pth_damage)
pth_constitutive.add_script(pth_fenics_mechanics)

class NormVM:
    def __init__(self, constraint, stress_norm):
        self.constraint = constraint
        self.ss_dim = ss_dim(self.constraint)
        self.stress_norm = stress_norm # True: norm of stress, False: norm of strain
        
        if _ss_vector == 'Voigt':
            if self.stress_norm:
                _fact = 6.0
            else: # for strain
                _fact = 1.5
        elif _ss_vector == 'Mandel':
            _fact = 3.0
        
        if self.ss_dim == 1:
            self.P = np.array([2/3])
        elif self.ss_dim==3:
            self.P = (1/3) * np.array([[2, -1, 0], [-1, 2, 0], [0, 0, _fact]])
        elif self.ss_dim==4:
            self.P = (1/3) * np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 2, 0], [0, 0, 0, _fact]])
        elif self.ss_dim==6:
            self.P = (1/3) * np.array( [ [2, -1, -1, 0, 0, 0], [-1, 2, -1, 0, 0, 0], [-1, -1, 2, 0, 0, 0] \
                                       , [0, 0, 0, _fact, 0, 0], [0, 0, 0, 0, _fact, 0], [0, 0, 0, 0, 0, _fact] ] )
    def __call__(self, ss):
        """
        ss: stress/strain vector for which
        se: the VM norm
        and
        m: the derivative of se w.r.t. ss
        are computed/returned.
        """
        if self.ss_dim==1:
            se = abs(ss)
            m = 1 # default for the case _norm=0
            if se>0:
                m = np.sign(ss[0])
        else:
            assert (len(ss)==self.ss_dim)
            se = (1.5 * ss.T @ self.P @ ss) ** 0.5
            if se == 0:
                m = np.zeros(self.ss_dim)  # using central difference method
                # m = np.sqrt(1.5 * np.diag(self.P)) # using forward difference method
            else:
                m = (1.5 / se) * self.P @ ss
        return se, m

class ElasticConstitutive():
    def __init__(self, E, nu, constraint):
        self.E = E
        self.nu = nu
        self.constraint = constraint
        self.ss_dim = ss_dim(self.constraint)
        
        if self.ss_dim == 1:
            self.dim = 1
            self.D = np.array([self.E])
        else:
            self.mu, self.lamda = constitutive_coeffs(E=self.E, nu=self.nu, constraint=constraint)
            if _ss_vector == 'Voigt':
                _fact = 1
            elif _ss_vector == 'Mandel':
                _fact = 2
            if self.ss_dim == 3:
                self.dim = 2
                self.D = (self.E / (1 - self.nu ** 2)) * np.array([ [1, self.nu, 0], [self.nu, 1, 0], [0, 0, _fact * 0.5 * (1-self.nu) ] ])
            elif self.ss_dim == 4:
                self.dim = 2
                self.D = np.array([
                        [2 * self.mu + self.lamda, self.lamda, self.lamda, 0],
                        [self.lamda, 2 * self.mu + self.lamda, self.lamda, 0],
                        [self.lamda, self.lamda, 2 * self.mu + self.lamda, 0],
                        [0, 0, 0, _fact * self.mu],
                    ])
            elif self.ss_dim == 6:
                self.dim = 3
                self.D = np.array([
                        [2 * self.mu + self.lamda, self.lamda, self.lamda, 0, 0, 0],
                        [self.lamda, 2 * self.mu + self.lamda, self.lamda, 0, 0, 0],
                        [self.lamda, self.lamda, 2 * self.mu + self.lamda, 0, 0, 0],
                        [0, 0, 0, _fact * self.mu, 0, 0],
                        [0, 0, 0, 0, _fact * self.mu, 0],
                        [0, 0, 0, 0, 0, _fact * self.mu],
                    ])
    
    def sigma_of_eps(self, eps, K=None):
        if self.ss_dim == 1:
            return self.E * eps
        else:
            return self.lamda * df.tr(eps) * df.Identity(self.dim) + 2 * self.mu * eps
    
    def sigma(self, u, K=None):
        return self.sigma_of_eps(eps=epsilon(u, self.dim), K=K)

class GradientDamageConstitutive(ElasticConstitutive):
    
    def __init__(self, E, nu, constraint, gK, c, h=0.0, interaction_function=lambda D:1, fn_eps_eq=None, epsilon_eq_num=None):
        super().__init__(E, nu, constraint)
        
        self.gK = gK
        self.c = c # gradient damage constant (l^2 in Poh's 2017 paper)
        self.h = h # coupling modulus (of Poh's paper in 2017)
        self.interaction_function = interaction_function # introduced as "g" in Poh's paper in 2017
        ### NOTE: The default values of "h" and "interaction_function" imply the "conventional gradient damage" introduced in Peerling's paper in 1996.
        if fn_eps_eq is None:
            if self.ss_dim == 1:
                self.fn_eps_eq = eps_eq_1d
                self.fn_d_eps_eq_d_eps = lambda x: ufl.operators.sign(x)
            else:
                self.fn_eps_eq = ModifiedMises(nu=self.nu)
                self.fn_d_eps_eq_d_eps = self.fn_eps_eq.d__d_eps
        else:
            self.fn_eps_eq = fn_eps_eq
        ### IMPORTANT: The "self.fn_eps_eq" method must get a "eps_3d" as its input argument
        
        self.epsilon_eq_num = epsilon_eq_num # a callable with argument of epsilon (of type numpy), which returns the equivalent strain and its derivative w.r.t. the given epsilon.
    
    def sigma_of_eps(self, eps, K):
        if self.ss_dim == 1:
            return (1 - self.gK.g(K)) * self.E * eps
        else:
            return (1 - self.gK.g(K)) * (self.lamda * df.tr(eps) * df.Identity(self.dim) + 2 * self.mu * eps)
        
    def d_sigma_d_D(self, u, K):
        eps_u = epsilon(u, self.dim)
        if self.ss_dim == 1:
            return - self.E * eps_u
        else:
            return - (self.lamda * df.tr(eps_u) * df.Identity(self.dim) + 2 * self.mu * eps_u)

    ## NOTE: In FEniCS generally, we can solve the problem without needing to define "d_sigma": the derivative of stress by ourselves. (The nonlinear variational solver can perform it by itself)
    
    def sigma_coupling(self, u, ebar): ### Not often used, thus, not yet implemented completely.
        """
        The last term in eq. (8) of the Poh's paper [2017]
        """
        if self.h==0:
            return 0.0
        else:
            return None ###################### To be developed        
    
    def epsilon_eq(self, epsilon):  # equivalent strain
        e_3d = eps_3d(epsilon, self.nu, self.constraint)
        return self.fn_eps_eq(e_3d)
    
    def update_K(K_old, ebar):
        """
        A static function to update internal variables.
        "K_old" and "ebar" are both FEniCS functions.
        This basically returns maximum of the two functions.
        """
        # ## WAY 1 (using ufl.conditional)
        # condition = ufl.le(K_old, ebar)
        # true_value = ebar
        # false_value = K_old
        # return ufl.conditional(condition, true_value, false_value)
        
        ## WAY 2 (using ufl.operators.Max)
        return ufl.operators.Max(K_old, ebar)
        
        # ## WAY 3 (using direct algebraic calculation)
        # return (K_old + ebar + ufl.algebra.Abs(K_old - ebar)) / Constant(2)
        
        ## WAY 4 (vector-based) (Completely different)
        # K_old_vec = K_old.vector().get_local()
        # ebar_vec = ebar.vector().get_local()
        # f_space = K_old.function_space()
        # K_new = Function(f_space)
        # K_new.vector().set_local(np.maximum(K_old_vec, ebar_vec))
        # return K_new
        
    def d_K_d_K_old(K_old, ebar):
        ## WAY 1 (using ufl.conditional)
        condition = ufl.ge(K_old, ebar)
        true_value = 1
        false_value = 0
        return ufl.conditional(condition, true_value, false_value)