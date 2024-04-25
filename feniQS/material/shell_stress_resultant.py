import numpy as np
from feniQS.general.general import CollectPaths

pth_shell_stress_resultant = CollectPaths('./feniQS/material/shell_stress_resultant.py')

class StressNormIlyushin:
    A = np.array([[1., -0.5, 0.], [-0.5, 1., 0.], [0., 0., 3.]])
    def __init__(self, coupling=True, alpha=0.25, y0=1.0, thickness=1.0):
        """
        This class concerns the so-called "Ilyushin" norm of stress resultant components
        , which is relevant for the perfect plasticity.
        If coupling==True, the coupling between membrane and bending is considered, otherwise not.
        """
        self.alpha = alpha
        self.y0 = y0 # uniaxial yield stress
        self.thickness = thickness
        self.coupling = coupling
        self._n0, self._m0, self._q0 = StressNormIlyushin.yield_stress_resultants(
            y0=self.y0, thickness=self.thickness)
        self._Ay_1 = StressNormIlyushin.yield_matrix(s=1. \
                    , _n0=self._n0, _m0=self._m0, _q0=self._q0 \
                    , coupling=self.coupling)
        self._Ay_2 = StressNormIlyushin.yield_matrix(s=-1. \
                    , _n0=self._n0, _m0=self._m0, _q0=self._q0 \
                    , coupling=self.coupling)

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
        if thickness is None:
            thickness = self.thickness
        ## (1)
        s_eq = (Nx**2 + Ny**2 - Nx*Ny + 3.*(Nxy**2)) / (thickness**2)
        s_eq += (Mx**2 + My**2 - Mx*My + 3.*(Mxy**2)) / (self.alpha**2) / (thickness**4)
        s_eq += 3. * (qxz**2 + qyz**2) / (thickness**2)
        if self.coupling:
            P = Nx*Mx + Ny*My - 0.5*Nx*My - 0.5*Ny*Mx + 3.*Nxy*Mxy
            s = 1 if P==0. else np.sign(P)
            s_eq += s * P / (np.sqrt(3.) * self.alpha * (thickness**3))
        else:
            s = None # irrelevant
        # (2) Using "_Ay_s" matrix (by y0=1.)
        if thickness==self.thickness and self.y0==1.0:
            _Ay_s = self._Ay_1 if s==1 else self._Ay_2
        else:
            _n0, _m0, _q0 = StressNormIlyushin.yield_stress_resultants(y0=1., thickness=thickness)
            _Ay_s = StressNormIlyushin.yield_matrix(s=s, _n0=_n0, _m0=_m0, _q0=_q0, coupling=self.coupling)
        s_vector = np.array([Nx, Ny, Nxy, Mx, My, Mxy, qxz, qyz]).reshape((-1,1))
        s_eq_2 = s_vector.T @ _Ay_s @ s_vector
        assert abs(s_eq - s_eq_2) / max(1., abs(s_eq)) < 1e-12
        return np.sqrt(s_eq)

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
        if thickness is None:
            thickness = self.thickness
        if thickness==self.thickness and self.y0==1.0:
            _Ay_1 = self._Ay_1
            _Ay_2 = self._Ay_2
        else:
            _n0, _m0, _q0 = StressNormIlyushin.yield_stress_resultants(y0=1., thickness=thickness)
            _Ay_1 = StressNormIlyushin.yield_matrix(s=1, _n0=_n0, _m0=_m0, _q0=_q0, coupling=self.coupling)
            _Ay_2 = StressNormIlyushin.yield_matrix(s=-1, _n0=_n0, _m0=_m0, _q0=_q0, coupling=self.coupling)
        s_vector = np.array([Nx, Ny, Nxy, Mx, My, Mxy, qxz, qyz]).reshape((-1,1))
        s_eq_1 = (s_vector.T @ _Ay_1 @ s_vector)[0,0]
        s_eq_2 = (s_vector.T @ _Ay_2 @ s_vector)[0,0]
        return np.sqrt(s_eq_1), np.sqrt(s_eq_2)
    
    @staticmethod
    def yield_stress_resultants(y0, thickness):
        _n0 = thickness * y0
        _m0 = (thickness**2) * y0 / 4.
        _q0 = thickness * y0 / np.sqrt(3.)
        return _n0, _m0, _q0
    
    @staticmethod
    def yield_matrix(s, _n0, _m0, _q0=None, coupling=True):
        if coupling:
            assert (s==1. or s==-1.)
            b12 = s / (2.*np.sqrt(3.)*_n0*_m0)
        else:
            b12 = 0.
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