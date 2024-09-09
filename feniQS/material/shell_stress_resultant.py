import numpy as np
from feniQS.general.general import CollectPaths
from feniQS.material.fenics_mechanics import *

pth_shell_stress_resultant = CollectPaths('./feniQS/material/shell_stress_resultant.py')
pth_shell_stress_resultant.add_script(pth_fenics_mechanics)

class StressNormIlyushin:
    A = np.array([[1., -0.5, 0.], [-0.5, 1., 0.], [0., 0., 3.]])
    def __init__(self, coupling=True, alpha=0.25, y0=1.0, thickness=1.0):
        """
        This class concerns the so-called "Ilyushin" norm of stress resultant components
        , which is relevant for the perfect plasticity.
        If coupling==True, the coupling between membrane and bending is considered.
        If coupling==False, we have the simplified Ilyshuin (s=0).
        The method "self.eq_stress_single" always computes the "un-coupled" case (s=0)
            , and if self.coupling==True it additionally computes the equivalent stress
            with coupling terms.
        """
        self.alpha = alpha
        self.y0 = y0 # uniaxial yield stress
        self.thickness = thickness
        self.coupling = coupling
        self._n0, self._m0, self._q0 = StressNormIlyushin.yield_stress_resultants(
            y0=self.y0, thickness=self.thickness)
        self._Ay_0 = StressNormIlyushin.yield_matrix(s=0. \
                    , _n0=self._n0, _m0=self._m0, _q0=self._q0) # corresponds to "NO coupling"
        if self.coupling:
            self._Ay_1 = StressNormIlyushin.yield_matrix(s=1. \
                        , _n0=self._n0, _m0=self._m0, _q0=self._q0)
            self._Ay_2 = StressNormIlyushin.yield_matrix(s=-1. \
                        , _n0=self._n0, _m0=self._m0, _q0=self._q0)

    def eq_equivalent_N(self, Nx, Ny, Nxy, normalize=True, thickness=None):
        if thickness is None:
            thickness = self.thickness
        N_eq = np.sqrt(Nx**2 + Ny**2 - Nx*Ny + 3.*(Nxy**2))
        if normalize: # Although self.y0 is by default 1, it still makes sense to normalize N_eq (to account for the effect of thickness).
            return N_eq / self._n0
        else:
            return N_eq
    
    def eq_equivalent_M(self, Mx, My, Mxy, normalize=True, thickness=None):
        if thickness is None:
            thickness = self.thickness
        M_eq = np.sqrt(Mx**2 + My**2 - Mx*My + 3.*(Mxy**2))
        if normalize: # Although self.y0 is by default 1, it still makes sense to normalize M_eq (to account for the effect of thickness).
            return M_eq / self._m0
        else:
            return M_eq
    
    def eq_stress_single(self, Nx, Ny, Nxy, Mx, My, Mxy, qxz=0., qyz=0. \
                         , thickness=None):
        """
        Inputs are:
            Nx, Ny, Nxy: membrane forces,
            Mx, My, Mxy: bending moments,
            qxz, qyz: shear forces,
        , which are NOT of stress unit. The corresponding stress values are termed as
        "generalized stresses" in the literature, which are ('t' stands for thickness):
            Nx/t, Ny/t, Nxy/t: generalized membrane stresses,
            Mx/(t^2), My/(t^2), Mxy/(t^2): generalized bending stresses,
            qxz/t, qyz/: generalized shear stresses.

        This method computes equivalent stress (single output) based on:
            Eq.(9) (with arbitrary alpha) in:
                https://www.sciencedirect.com/science/article/pii/S0045794901000323
            or:
            Eq.(64) (for alpha=0.25) in:
                https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0207(19991230)46:12%3C1961::AID-NME759%3E3.0.CO;2-E
        """
        if np.isscalar(Nx):
            Nx = np.array([Nx])
        if np.isscalar(Ny):
            Ny = np.array([Ny])
        if np.isscalar(Nxy):
            Nxy = np.array([Nxy])
        if np.isscalar(Mx):
            Mx = np.array([Mx])
        if np.isscalar(My):
            My = np.array([My])
        if np.isscalar(Mxy):
            Mxy = np.array([Mxy])
        if np.isscalar(qxz):
            qxz = np.array([qxz])
        if np.isscalar(qyz):
            qyz = np.array([qyz])
        assert isinstance(Nx, np.ndarray)
        _size = len(Nx)
        assert all([len(a)==_size for a in [Ny, Nxy, Mx, My, Mxy, qxz, qyz]])
        
        if thickness is None:
            thickness = self.thickness
        
        ## (1)
        s_eq0 = (Nx**2 + Ny**2 - Nx*Ny + 3.*(Nxy**2)) / (thickness**2)
        s_eq0 += (Mx**2 + My**2 - Mx*My + 3.*(Mxy**2)) / (self.alpha**2) / (thickness**4)
        s_eq0 += 3. * (qxz**2 + qyz**2) / (thickness**2)
        if self.coupling:
            P = Nx*Mx + Ny*My - 0.5*Nx*My - 0.5*Ny*Mx + 3.*Nxy*Mxy
            s = np.where(P == 0., 1, np.sign(P))
            s_eq = s_eq0 + s * P / (np.sqrt(3.) * self.alpha * (thickness**3))
        else:
            s_eq = s_eq0
        
        # (2) Using "_Ay_s" matrix (by y0=1.)
        s_vector = np.array([Nx, Ny, Nxy, Mx, My, Mxy, qxz, qyz])
            # With no coupling (s=0) is computed by default.
        if thickness==self.thickness and self.y0==1.0:
            _Ay_0 = self._Ay_0
        else:
            _n0, _m0, _q0 = StressNormIlyushin.yield_stress_resultants(y0=1., thickness=thickness)
            _Ay_0 = StressNormIlyushin.yield_matrix(s=0, _n0=_n0, _m0=_m0, _q0=_q0)
        s_eq0_2 = np.einsum('mi,mn,ni->i', s_vector, _Ay_0, s_vector)
        if self.coupling:
            if thickness==self.thickness and self.y0==1.0:
                _Ay_s = np.array([self._Ay_1 if si==1 else self._Ay_2 for si in s])
            else:
                _n0, _m0, _q0 = StressNormIlyushin.yield_stress_resultants(y0=1., thickness=thickness)
                _Ay_s = np.array([StressNormIlyushin.yield_matrix(s=si, _n0=_n0, _m0=_m0, _q0=_q0) for si in s])
            s_eq_2 = np.einsum('mi,imn,ni->i', s_vector, _Ay_s, s_vector)
        else:
            s_eq_2 = s_eq0_2

        assert np.linalg.norm(s_eq0 - s_eq0_2) / max(1., np.linalg.norm(s_eq0)) < 1e-12
        assert np.linalg.norm(s_eq - s_eq_2) / max(1., np.linalg.norm(s_eq)) < 1e-12

        if _size==1:
            return np.sqrt(s_eq0)[0], np.sqrt(s_eq)[0]
        else:
            return np.sqrt(s_eq0), np.sqrt(s_eq)

    def eq_stresses_double(self, Nx, Ny, Nxy, Mx, My, Mxy, qxz=0., qyz=0. \
                         , thickness=None):
        """
        Inputs are:
            Nx, Ny, Nxy: membrane forces,
            Mx, My, Mxy: bending moments,
            qxz, qyz: shear forces,
        , which are NOT of stress unit. The corresponding stress values are termed as
        "generalized stresses" in the literature, which are ('t' stands for thickness):
            Nx/t, Ny/t, Nxy/t: generalized membrane stresses,
            Mx/(t^2), My/(t^2), Mxy/(t^2): generalized bending stresses,
            qxz/t, qyz/: generalized shear stresses.

        This method computes equivalent stresses (double outputs) based on:
            Eq.(6) (for alpha=0.25) in:
                https://www.sciencedirect.com/science/article/pii/S0263823116300118
        """
        if np.isscalar(Nx):
            Nx = np.array([Nx])
        if np.isscalar(Ny):
            Ny = np.array([Ny])
        if np.isscalar(Nxy):
            Nxy = np.array([Nxy])
        if np.isscalar(Mx):
            Mx = np.array([Mx])
        if np.isscalar(My):
            My = np.array([My])
        if np.isscalar(Mxy):
            Mxy = np.array([Mxy])
        if np.isscalar(qxz):
            qxz = np.array([qxz])
        if np.isscalar(qyz):
            qyz = np.array([qyz])
        assert isinstance(Nx, np.ndarray)
        _size = len(Nx)
        assert all([len(a)==_size for a in [Ny, Nxy, Mx, My, Mxy, qxz, qyz]])

        if thickness is None:
            thickness = self.thickness
        
        s_vector = np.array([Nx, Ny, Nxy, Mx, My, Mxy, qxz, qyz])
        if self.coupling:
            if thickness==self.thickness and self.y0==1.0:
                _Ay_1 = self._Ay_1
                _Ay_2 = self._Ay_2
            else:
                _n0, _m0, _q0 = StressNormIlyushin.yield_stress_resultants(y0=1., thickness=thickness)
                _Ay_1 = StressNormIlyushin.yield_matrix(s=1, _n0=_n0, _m0=_m0, _q0=_q0)
                _Ay_2 = StressNormIlyushin.yield_matrix(s=-1, _n0=_n0, _m0=_m0, _q0=_q0)
            s_eq_1 = np.einsum('mi,mn,ni->i', s_vector, _Ay_1, s_vector)
            s_eq_2 = np.einsum('mi,mn,ni->i', s_vector, _Ay_2, s_vector)
        else: # s=0
            if thickness==self.thickness and self.y0==1.0:
                _Ay_0 = self._Ay_0
            else:
                _n0, _m0, _q0 = StressNormIlyushin.yield_stress_resultants(y0=1., thickness=thickness)
                _Ay_0 = StressNormIlyushin.yield_matrix(s=0, _n0=_n0, _m0=_m0, _q0=_q0)
            s_eq_1 = s_eq_2 = np.einsum('mi,mn,ni->i', s_vector, _Ay_0, s_vector)
        
        if _size==1:
            return np.sqrt(s_eq_1)[0], np.sqrt(s_eq_2)[0]
        else:
            return np.sqrt(s_eq_1), np.sqrt(s_eq_2)
    
    @staticmethod
    def yield_stress_resultants(y0, thickness):
        _n0 = thickness * y0
        _m0 = (thickness**2) * y0 / 4.
        _q0 = thickness * y0 / np.sqrt(3.)
        return _n0, _m0, _q0
    
    @staticmethod
    def yield_matrix(s, _n0, _m0, _q0=None):
        assert (s==1. or s==-1. or s==0.) # The case "s=0" corresponds to NO coupling.
        b12 = s / (2.*np.sqrt(3.)*_n0*_m0)
        B = np.array([[_n0**(-2), b12],
                      [b12, _m0**(-2)]])
        Ay = np.kron(B, StressNormIlyushin.A)
        if _q0 is not None:
            qy = (_q0**(-2)) * np.eye(2)
            Ay = np.block([[Ay, np.zeros((Ay.shape[0], 2))],
              [np.zeros((2, Ay.shape[1])), qy]])
        return Ay
    
    @staticmethod
    def get_generalized_stresses(resultant_components, thickness):
        Nx, Ny, Nxy, Mx, My, Mxy, qxz, qyz = resultant_components # unpack
        t2 = thickness**2
        return [Nx/thickness, Ny/thickness, Nxy/thickness,
                Mx/t2, My/t2, Mxy/t2,
                qxz/thickness, qyz/thickness]

    @staticmethod
    def get_resultant_components(generalized_stresses, thickness):
        gNx, gNy, gNxy, gMx, gMy, gMxy, gqxz, gqyz = generalized_stresses # unpack
        t2 = thickness**2
        return [gNx*thickness, gNy*thickness, gNxy*thickness,
                gMx*t2, gMy*t2, gMxy*t2,
                gqxz*thickness, gqyz*thickness]

class ElasticShellStressResultantConstitutive():
    def __init__(self, E, nu, thickness \
                , constraint='PLANE_STRESS' \
                , shear_correction_factor=1.):
        self.E = E
        self.nu = nu
        self.thickness = thickness # default (uniform) thickness
        if constraint!='PLANE_STRESS':
            raise NotImplementedError(f"Elastic shell stress resultant is only implemented for PLANA_STRESS.")
        self.constraint = constraint
        self.shear_correction_factor = shear_correction_factor
        self.dim = 2
        self.ss_dim = 3
        self.ss_vector = 'Voigt'; _fact = 1
        self.mu, self.lamda = constitutive_coeffs(E=self.E, nu=self.nu, constraint=self.constraint)
        self.D = (self.E / (1 - self.nu ** 2)) * np.array([ [1, self.nu, 0], [self.nu, 1, 0], [0, 0, _fact * 0.5 * (1-self.nu) ] ])

        self._C_membrane = self.thickness * self.D
        self._C_bending = (self.thickness ** 3) * self.D / 12.
        self._C_shear = self.shear_correction_factor * self.thickness * self.mu * np.eye(2)
    
    def get_C_membrane(self, thickness=None):
        thickness = self.thickness if thickness is None else thickness
        if thickness==self.thickness:
            return self._C_membrane
        else:
            return thickness * self.D
    
    def get_C_bending(self, thickness=None):
        thickness = self.thickness if thickness is None else thickness
        if thickness==self.thickness:
            return self._C_bending
        else:
            return (self.thickness ** 3) * self.D / 12.
    
    def get_C_shear(self, thickness=None):
        thickness = self.thickness if thickness is None else thickness
        if thickness==self.thickness:
            return self._C_shear
        else:
            return self.shear_correction_factor * self.thickness * self.mu * np.eye(2)