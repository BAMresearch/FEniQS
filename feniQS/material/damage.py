import numpy as np
import dolfin as df
import ufl

from feniQS.general.general import CollectPaths
from feniQS.fenics_helpers.fenics_functions import conditional_by_ufl, pth_fenics_functions

pth_damage = CollectPaths('./feniQS/material/damage.py')
pth_damage.add_script(pth_fenics_functions)

class GKLocalDamageLinear: ### Developed now for numpy usage (NOT dolfin Ufl)
    """
    To define a linear damage law defined as:
        g(K) = (ku / (ku - ko)) * (1. - ko / K)
        , with ko, ku being constant values.
    """

    def __init__(self, ko=0.0001, ku=0.0125):
        self.ko = ko
        self.ku = ku
        self._fact = self.ku / (self.ku - self.ko) # as a helper value
    
    def g(self, K):
        vt = 0.
        vf = self._fact * (1.0 - self.ko / K)
        return conditional_by_ufl(condition='le', f1=K, f2=self.ko \
                                  , value_true=vt, value_false=vf)
    
    def g_eval(self, K_val):
        if K_val <= self.ko:
            return 0.0
        else:
            return self._fact * (1.0 - self.ko / K_val)
    
    def dg_dK_eval(self, K_val):
        if K_val <= self.ko:
            return 0.0
        else:
            return self._fact * self.ko / (K_val**2)

class GKLocalDamageExponential:
    """
    To define exponential damage law "g(K)" according to the eq.(5.39) of Milan Jirasek lecture book and also eq. (15) of the Poh's [2017] paper.
        ---> The former is an especial case of the latter for alpha=1.
        
    Also, we have always:
        K0=e0
        beta=1/ef
        So, the class is developed based on "e0" and "ef" as input arguments.
    """

    def __init__(self, e0=0.03, ef=0.05, alpha=1.):
        self.alpha = alpha
        self.e0 = e0
        self.ef = ef
        
    def g(self, K):
        vt = 0.
        vf = 1 - self.e0 * (1 - self.alpha + self.alpha * df.exp((self.e0 - K) / self.ef)) / K
        return conditional_by_ufl(condition='le', f1=K, f2=self.e0 \
                                  , value_true=vt, value_false=vf)
    
    def dg_dK(self, K):
        vt = 0.
        vf = self.e0 / K * ((1.0 / K + 1.0/self.ef) * self.alpha * df.exp((self.e0 - K) / self.ef) + (1.0 - self.alpha) / K)
        return conditional_by_ufl(condition='le', f1=K, f2=self.e0 \
                                  , value_true=vt, value_false=vf)
    
    def g_eval(self, K_val):
        """
        For simple evaluation at given K value
        """
        if type(self.e0)==float or type(self.e0)==int:
            if K_val <= self.e0:
                return 0
            else:
                return 1 - self.e0 * (1 - self.alpha + self.alpha * np.exp((self.e0 - K_val) / self.ef)) / K_val
        else:
            return None
    
    @staticmethod
    def extract_features_from_load_disp(us, fs, unloading_df_ratio=None):
        assert len(us) == len(fs)
        fs = np.array(fs)
        id_max_f = fs.argmax()
        f_max = max(abs(fs))
        slope_loading, b = np.polyfit(us[0:id_max_f], fs[0:id_max_f], 1)
        u_tip = us[id_max_f]
        n_unloading = len(fs) - id_max_f - 1
        if n_unloading > 0: # we have unloading as well
            # Default: unloading is considered based on the single data point right after f_max
            slope_unloading = (fs[id_max_f + 1] - fs[id_max_f]) / (us[id_max_f+1] - us[id_max_f])
            if unloading_df_ratio is not None:
                us_unloading = [us[id_max_f]]
                fs_unloading = [fs[id_max_f]]
                for ii, ff in enumerate(fs[id_max_f+1:]):
                    if abs(ff) >= unloading_df_ratio * f_max:
                        us_unloading.append(us[ii+id_max_f+1])
                        fs_unloading.append(ff)
                if len(us_unloading) > 2: # If ==2, then it is equivalent to the slope assigned above by default.
                    slope_unloading, b = np.polyfit(us_unloading, fs_unloading, 1)
        else:
            slope_unloading = None
        
        summary = f"slope_loading     = {slope_loading:.2e}\nf_max                 = {f_max:.2e}\nu_tip                   = {u_tip:.2e}"
        if slope_unloading is not None:
            summary += f"\nslope_unloading = {slope_unloading:.2e}"
        
        return slope_loading, id_max_f, u_tip, slope_unloading, summary
                

    def maximum_dissipation(self, E=1., e0=None, ef=None):
        """
        Integral of (1-g(K)) * E * K for K in [0, +inf]
        """
        if e0 is None:
            e0 = self.e0
        if isinstance(e0, df.Constant):
            e0 = e0.values()[0]
        if ef is None:
            ef = self.ef
        if isinstance(ef, df.Constant):
            ef = ef.values()[0]
        d = E * e0 * (0.5 * e0 + ef)
        if self.alpha!=1.0:
            print(f"WARNING: Computed maximum energy dissipation is not exact, since alpha!=1.0 .")
        return d

class GKLocalDamagePerfect:
    """
    To define damage law:
        g(K) and dg_dK(K),
    which will give the following perfectly-damaged strain-stress curve (turning point at strain=K0):
    
    Stress
    ^
    |
    |
   E*K0   ______________________
    |    /
    |   /
    |  /
    | /
    |/
     _____K0_________________________>  Strain
    
    """

    def __init__(self, K0=0.03):
        self.K0 = K0

    def g(self, K):
        vt = 0.
        vf = 1. - self.K0 / K
        return conditional_by_ufl(condition='le', f1=K, f2=self.K0 \
                                  , value_true=vt, value_false=vf)
    
    def dg_dK(self, K):
        vt = 0.
        vf = self.K0 / (K*K)
        return conditional_by_ufl(condition='le', f1=K, f2=self.K0 \
                                  , value_true=vt, value_false=vf)
    
    def g_eval(self, K_val):
        """
        For simple evaluation at given K value
        """
        if type(self.K0)==float or type(self.K0)==int:
            if K_val <= self.K0:
                return 0.0
            else:
                return 1.0 - self.K0 / K_val
        else:
            return None
    
    def dg_dK_eval(self, K_val):
        if K_val <= self.K0:
            return 0.0
        else:
            return self.K0 / (K_val**2)

    def maximum_dissipation(self, E=1.0):
        """
        Integral of (1-g(K)) * E * K for K in [0, infty]
        """
        return float("inf")
    
class InteractionFunction_exp():
    """
    Equation (16) of Poh's paper [2017]
    """
    def __init__(self, eta=5, R=0.005):
        self.eta = eta
        self.R = R
    
    def __call__(self, D):
        return ( (1 - self.R) * ufl.operators.exp(-self.eta * D) + self.R - ufl.operators.exp(-self.eta) ) / (1 - ufl.operators.exp(-self.eta))