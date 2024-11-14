import numpy as np
from feniQS.material.constitutive import *

pth_constitutive_plastic = CollectPaths('./feniQS/material/constitutive_plastic.py')
pth_constitutive_plastic.add_script(pth_constitutive)

class RateIndependentHistory:
    def __init__(self, p=None, dp_dsig=None, dp_dk=None):
        """
        The goal of this class is to hardcode the consistent provide of enough information about a rate-independent history evolution.
        This includes a callable (p_and diffs) with:
            INPUTS:
                sigma: effective (elastic) stress
                kappa: internal variable(s)
            OUTPUTS:
                p: (depending on material) the intended rate-independent function "p(sigma, kappa)"
                    , such that: kappa_dot = lamda_dot * p , where lamda is the well-known "plastic multiplier".
                    ---> this is a "rate-independent" evolution of "kappa".
                dp_dsig: accordingly the derivative of "p" w.r.t. sigma (with same inputs)
                dp_dk: accordingly the derivative of "p" w.r.t. kappa (with same inputs)
        IMPORTANT:
            All outputs must be in form of 2-D np.array with a full shape (m,n), e.g. scalar values are of shape (1,1)
        """
        if p is None: # default is p=1, i.e. kappa_dot = lamda_dot
            def p_and_diffs(sigma, kappa):
                return np.array([[1.0]]), np.array([[0.0]]), np.array([[0.0]])
        else:
            assert(dp_dsig is not None)
            assert(dp_dk is not None)
            def p_and_diffs(sigma, kappa):
                return p(sigma, kappa), dp_dsig(sigma, kappa), dp_dk(sigma, kappa)
        self.p_and_diffs = p_and_diffs
    def __call__(self, sigma, kappa):
        """
        sigma: effective stress
        kappa: internal variable(s)
        returns:
            p, dp_dsig, dp_dk (all explained in constructor)
        """
        return self.p_and_diffs(sigma, kappa)

class RateIndependentHistory_PlasticWork(RateIndependentHistory):
    def __init__(self, yield_function):
        self.yf = yield_function
        def p_and_diffs(sigma, kappa):
            _, m, dm, _, mk = self.yf(sigma, kappa)
            p = np.array([[np.dot(sigma, m)]])
            dp_dsig = (m + dm.T @ sigma).reshape((1, -1))
            dp_dk = np.array([[np.dot(sigma, mk)]])
            return p, dp_dsig, dp_dk
        self.p_and_diffs = p_and_diffs
    def __call__(self, sigma, kappa):
        return self.p_and_diffs(sigma, kappa)

class PlasticConsitutivePerfect(ElasticConstitutive):
    def __init__(self, E, nu, constraint, yf):
        """
        yf: a callable with:
            argument:      stress state
            return values: f: yield surface evaluation
                           m: flow vector; e.g. with associated flow rule: the first derivative of yield surface w.r.t. stress
                           dm: derivative of flow vector w.r.t. stress; e.g. with associated flow rule: second derivative of yield surface w.r.t. stress
        """
        super().__init__(E=E, nu=nu, constraint=constraint)
        self.yf = yf
        assert(self.ss_dim == self.yf.ss_dim)
    
    def correct_stress(self, sig_tr, k0=0, _Ct=True, tol=1e-9, max_iters=20):
        """
        sig_tr: trial (predicted) stress
        k0: last converged internal variable(s), which is not relevant here, but is given just for the sake of generality
        returns:
            sig_c: corrected_stress
            dl: lamda_dot
            Ct: corrected stiffness matrix
            m: flow vector (e.g. normal vector to the yield surface, if self.yf is associative)
        """
        
        ### WAY 1 ### Using Jacobian matrix to solve residuals (of the return mapping algorithm) in a coupled sense
        f, m, dm, _, _ = self.yf(sig_tr)
        if f > 0: # do return mapping
            Ct = None
            _d = len(sig_tr)
            # initial values of unknowns (we add 0.0 to something to have a copy regardless of type (float or np.array))
            sig_c = sig_tr + 0.0
            dl = 0.0
            # compute residuals of return-mapping (backward Euler)
            d_eps_p = dl * m # change in plastic strain
            es = sig_c - sig_tr + self.D @ d_eps_p
            ef = f
            e_norm = np.linalg.norm(np.append(es, ef))
            _it = 0
            while e_norm > tol and _it<=max_iters:
                A1 = np.append( np.eye(_d) + dl * self.D @ dm, (self.D @ m).reshape((-1,1)), axis=1 )
                A2 = np.append(m, 0).reshape((1,-1))
                Jac = np.append(A1, A2, axis=0)
                dx = np.linalg.solve(Jac, np.append(es, ef))
                sig_c -= dx[0:_d]     
                dl -= dx[_d:]
                f, m, dm, _, _ = self.yf(sig_c)
                d_eps_p = dl * m # change in plastic strain
                es = sig_c - sig_tr + self.D @ d_eps_p
                ef = f
                e_norm = np.linalg.norm(np.append(es, ef))
                _it += 1
            # after converging return-mapping:
            if _Ct:
                A1 = np.append( np.eye(_d) + dl * self.D @ dm, (self.D @ m).reshape((-1,1)), axis=1 )
                A2 = np.append(m, 0).reshape((1,-1))
                Jac = np.append(A1, A2, axis=0)
                inv_Jac = np.linalg.inv(Jac)
                Ct = inv_Jac[np.ix_(range(_d), range(_d))] @ self.D
            return sig_c, Ct, k0, d_eps_p
        else: # still elastic zone
            return sig_tr, self.D, k0, 0.0
        
        ### WAY 2 ### Direct (decoupled) iteration
        # f, m, dm, _, _ = self.yf(sig_tr)
        # if f > 0: # do return mapping
        #     Ct = None
        #     _d = len(sig_tr)
        #     dl = f / (m.T @ self.D @ m)
        #     sig_c = sig_tr - dl * self.D @ m
        #     f, m, dm, _, _ = self.yf(sig_c)
        #     d_eps_p = dl * m # change in plastic strain
        #     r = sig_c - (sig_tr - self.D @ d_eps_p)
        #     _it = 0
        #     while (f > tol or np.linalg.norm(r) > tol) and _it<=max_iters:
        #         Q = np.eye(_d) + dl * self.D @ dm
        #         ddl = (f - m.T @ np.linalg.solve(Q, r)) / (m.T @ np.linalg.solve(Q, self.D @ m))
        #         dl += ddl
        #         sig_c = sig_c - np.linalg.solve(Q, r) - ddl * np.linalg.solve(Q, self.D @ m)
        #         f, m, dm, _, _ = self.yf(sig_c)
        #         d_eps_p = dl * m # change in plastic strain
        #         r = sig_c - (sig_tr - self.D @ d_eps_p)
        #         _it += 1
        #     # after converging return-mapping:
        #     if _Ct:
        #         Q = np.eye(_d) + dl * self.D @ dm
        #         R = np.linalg.inv(Q) @ self.D
        #         _fact = 1 / (m.T @ R @ m)
        #         Ct = R @ (np.eye(_d) - _fact * np.outer(m, m) @ R)
        #     return sig_c, Ct, k0, d_eps_p
        # else: # still elastic zone
        #     return sig_tr, self.D, k0, 0.0
        
        
class PlasticConsitutiveRateIndependentHistory(PlasticConsitutivePerfect):
    def __init__(self, E, nu, constraint, yf, ri):
        """
        ri: an instance of RateIndependentHistory representing evolution of history variables
        , which is based on:
            kappa_dot = lamda_dot * p(sigma, kappa), where "p" is a plastic modulus function
        """
        super().__init__(E, nu, constraint, yf)
        assert(isinstance(ri, RateIndependentHistory))
        self.ri = ri # ri: rate-independent
    
    def correct_stress(self, sig_tr, k0=0.0, _Ct=True, tol=1e-9, max_iters=20):
        """
        overwritten to the superclass'
        one additional equation to be satisfied is the rate-independent equation:
            kappa_dot = lamda_dot * self.ri.p
        , for the evolution of the history variable(s) k
        """
        
        ### Solve residuals (of the return mapping algorithm) equal to ZERO, in a coupled sense (using Jacobian matrix based on backward Euler)
        f, m, dm, fk, mk = self.yf(sig_tr, k0)
        if f > 0: # do return mapping
            Ct = None
            _d = len(sig_tr)
            # initial values of unknowns (we add 0.0 to something to have a copy regardless of type (float or np.array))
            sig_c = sig_tr + 0.0
            k = k0 + 0.0
            dl = 0.0
            # compute residuals of return-mapping (backward Euler)
            d_eps_p = dl * m # change in plastic strain
            es = sig_c - sig_tr + self.D @ d_eps_p
            p, dp_dsig, dp_dk = self.ri(sig_c, k)
            if max(dp_dsig.shape) != _d:
                dp_dsig = dp_dsig * np.ones((1, _d))
            ek = k - k0 - dl * p
            ef = f
            e_norm = np.linalg.norm(np.append(np.append(es, ek), ef))
            _it = 0
            while e_norm > tol and _it<=max_iters:
                A1 = np.append( np.append(np.eye(_d) + dl * self.D @ dm, dl * (self.D @ mk).reshape((-1,1)), axis=1) \
                               , (self.D @ m).reshape((-1,1)), axis=1 )
                A2 = np.append(np.append(- dl * dp_dsig, 1 - dl * dp_dk, axis=1), -p, axis=1)
                A3 = np.append(np.append(m, fk), 0).reshape((1,-1))
                Jac = np.append(np.append(A1, A2, axis=0), A3, axis=0)
                dx = np.linalg.solve(Jac, np.append(np.append(es, ek), ef))
                sig_c -= dx[0:_d]     
                k -= dx[_d:_d+1]
                dl -= dx[_d+1:]
                f, m, dm, fk, mk = self.yf(sig_c, k)
                d_eps_p = dl * m # change in plastic strain
                es = sig_c - sig_tr + self.D @ d_eps_p
                p, dp_dsig, dp_dk = self.ri(sig_c, k)
                if max(dp_dsig.shape) != _d:
                    dp_dsig = np.zeros((1, _d))
                ek = k - k0 - dl * p
                ef = f
                e_norm = np.linalg.norm(np.append(np.append(es, ek), ef))
                _it += 1
            # after converging return-mapping:
            if _Ct:
                A1 = np.append( np.append(np.eye(_d) + dl * self.D @ dm, dl * (self.D @ mk).reshape((-1,1)), axis=1) \
                               , (self.D @ m).reshape((-1,1)), axis=1 )
                A2 = np.append(np.append(- dl * dp_dsig, 1 - dl * dp_dk, axis=1), -p, axis=1)
                A3 = np.append(np.append(m, fk), 0).reshape((1,-1))
                Jac = np.append(np.append(A1, A2, axis=0), A3, axis=0)
                inv_Jac = np.linalg.inv(Jac)
                Ct = inv_Jac[np.ix_(range(_d), range(_d))] @ self.D
            return sig_c, Ct, k, d_eps_p
        else: # still elastic zone
            return sig_tr, self.D, k0, 0.0
                
class Yield_VM:
    def __init__(self, y0, constraint, hardening_isotropic_law=None):
        self.y0 = y0 # yield stress
        self.constraint = constraint
        self.ss_dim = ss_dim(self.constraint)
        self.vm_norm = NormVM(self.constraint, stress_norm=True)
        
        if hardening_isotropic_law is None: # perfect plasticity
            hardening_isotropic_law = {'modulus': 0.}
        else:
            modulus = hardening_isotropic_law['modulus']
            if modulus!=0.: # isotropic hardening
                assert modulus>0.
                assert 'law' in hardening_isotropic_law.keys()
                if hardening_isotropic_law['law']=='linear':
                    _b = not 'sig_u' in hardening_isotropic_law.keys()
                    _b = _b or (hardening_isotropic_law['sig_u'] is None)
                    if _b:
                        hardening_isotropic_law['sig_u'] = np.inf
                        self._K0 = np.inf
                    else:
                        self._K0 = (hardening_isotropic_law['sig_u'] - self.y0) / modulus
                        self._modulus2 = modulus/100.
                            # : to avoid convergence issues, we set much smaller hardening modulus after ultimate strength is met.
                elif hardening_isotropic_law['law']=='exponential':
                    assert 'sig_u' in hardening_isotropic_law.keys() # and msut be a real number.
                    self._dy = hardening_isotropic_law['sig_u'] - self.y0
                else:
                    raise ValueError(f"Isotropic hardening law is not recognized.")
                assert hardening_isotropic_law['sig_u']>self.y0
        self.ihl = hardening_isotropic_law
        
    def __call__(self, stress, kappa=0):
        """
        Evaluate the yield function quantities at a specific stress level (as a vector):
            f: yield function itself
            m: flow vector; derivative of "f" w.r.t. stress (associated flow rule)
            dm: derivative of flow vector w.r.t. stress; second derivative of "f" w.r.t. stress
            fk: derivative of "f" w.r.t. kappa
            mk: derivative of "m" w.r.t. kappa
        The given stress vector must be consistent with self.ss_dim
        kappa: history variable(s), here related to isotropic hardening
        """
        assert (len(stress) == self.ss_dim)
        se, m = self.vm_norm(stress)

        if self.ihl['modulus']==0.:
            f = se - self.y0
            fk = 0.
        else:
            if self.ihl['law']=='linear':
                if kappa>self._K0:
                    kappa2 = kappa - self._K0
                    _y0 = self.ihl['sig_u'] + self._modulus2 * kappa2
                    f = se - _y0
                    fk = - self._modulus2
                else:
                    f = se - (self.y0 + self.ihl['modulus'] * kappa)
                    fk = - self.ihl['modulus']
            elif self.ihl['law']=='exponential':
                _exp = np.exp(-self.ihl['modulus']*kappa)
                f = se - self.y0 - self._dy * (1. - _exp)
                fk = - self._dy * self.ihl['modulus'] * _exp

        if self.ss_dim==1:
            dm = 0.0 # no dependency on any hardening modulus
            mk = 0.0
        else:
            if se ==0:
                dm = None # no needed in such a case
            else:
                dm = (6 * se * self.vm_norm.P - 6 * np.outer(self.vm_norm.P @ stress, m)) / (4 * se ** 2) # no dependency on any hardening modulus
            mk = np.array(len(stress) * [0.0])
        return f, np.atleast_1d(m), np.atleast_2d(dm), fk, np.atleast_1d(mk)
