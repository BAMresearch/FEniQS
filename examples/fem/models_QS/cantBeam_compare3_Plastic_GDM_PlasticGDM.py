from examples.fem.models_QS.cantileverBeam_plasticity_model import *
from examples.fem.models_QS.cantileverBeam_plasticGDM_model import *

class UnifiedParsCantileverBeamPlasticGDM:
    def __init__(self):
        self.constraint = 'PLANE_STRESS'
        # self.constraint = '3D'
        
        self.E = 1000.
        self.nu = 0.3
        self.lx = 6.0
        self.ly = 0.5
        self.lz = 0.5 # for ploting reaction force proportional to the lz
        
        self.unit_res = 6
        
        self.loading = ParsLoading(level=1)
            # ramp
        # self.loading.case = 'ramp'
        # self.loading.level = 4 * self.ly
        # self.loading.scales = np.array([1])
            # sinusoidal
        self.loading.N = 1.35
        self.loading.case = 'sin'
        self.loading.level = 2 * self.ly
        self.loading.scales = np.array([1])
        
        self.sol_tol = 1e-8
        if self.loading.case == 'ramp':
            self.sol_res = 101
        else:
            self.sol_res = int(self.loading.N * 180)
        
        # body force (zero)
        if self.constraint=='PLANE_STRESS' or self.constraint=='PLANE_STRAIN':
            self.geo_dim = 2
        elif self.constraint=='3D':
            self.geo_dim = 3
        self.f = df.Constant(self.geo_dim * [0.0])
        
        # others
        self._plot = True
        self._write_files = True
        
        # self.damage_type = 'perfect'
        self.damage_type = 'linear'
        
        self.e0 = 3e-3
        self.ef = 2e-1
        self.alpha = 0.99
        self.c = (2 / self.unit_res) ** 2
        
        # yield strength
        self.sig0 = 12.0
        
        ### NO Hardening
        # self.H = 0.0 # Hardening modulus
        
        ### ISOTROPIC Hardening
        Et = self.E / 100.0
        self.H = 15 * self.E * Et / (self.E - Et) # Hardening modulus
        ## Hardening hypothesis
        self.hardening_hypothesis = 'unit' # denoting the type of the harding function "P(sigma, kappa)"
        # self.hardening_hypothesis = 'plastic-work'
    
def run_3_simulations():
    df.set_log_level(30)
    
    pars = UnifiedParsCantileverBeamPlasticGDM()
    checkpoints = []
    
    ## Pure plasticity
    model1 = CantileverBeamPlasticModel(pars)
    reaction_dofs1 = [model1.fen.bcs_DR_inhom_dofs]
    pps1, plots_data1 = model1.solve(pars.loading.t_end, reaction_dofs=reaction_dofs1, checkpoints=checkpoints)
    rf1 = pps1[0].reaction_forces[0]
    msg1 = f"F_min, F_max = {min([sum(ri) for ri in rf1])}, {max([sum(ri) for ri in rf1])} ."
    model1.logger.file.debug('\n' + msg1)
    print(msg1)
    
    ## Pure GDM (by putting a very high value to pars.sig0)
    sig0_original = pars.sig0
    pars.sig0 = 1e13 # to have pure damage
    model2 = CantileverBeamPlasticGDMModel(pars, _name='CantBeamGDM_deg' + str(FenicsConfig.shF_degree_u) + '_' + str(pars.constraint))
    reaction_dofs2 = [model2.fen.bcs_DR_inhom_dofs]
    pps2, plots_data2 = model2.solve(pars.loading.t_end, reaction_dofs=reaction_dofs2, checkpoints=checkpoints)
    rf2 = pps2[0].reaction_forces[0]
    msg2 = f"F_min, F_max = {min([sum(ri) for ri in rf2])}, {max([sum(ri) for ri in rf2])} ."
    model2.logger.file.debug('\n' + msg2)
    print(msg2)
    
    ## GDM + Plasticity
    pars.sig0 = sig0_original
    model3 = CantileverBeamPlasticGDMModel(pars)
    reaction_dofs3 = [model3.fen.bcs_DR_inhom_dofs]
    pps3, plots_data3 = model3.solve(pars.loading.t_end, reaction_dofs=reaction_dofs3, checkpoints=checkpoints)
    rf3 = pps3[0].reaction_forces[0]
    msg3 = f"F_min, F_max = {min([sum(ri) for ri in rf3])}, {max([sum(ri) for ri in rf3])} ."
    model3.logger.file.debug('\n' + msg3)
    print(msg3)
    
    xs = plots_data1[0] + [plots_data2[0][1]] + [plots_data3[0][1]]
    ys = plots_data1[1] + [plots_data2[1][1]] + [plots_data3[1][1]]
    labels = plots_data1[2] + [plots_data2[2][1]] + [plots_data3[2][1]]
    labels[-2] = 'FEniCS: GDM' # pure damage
    
    ## Common plot of reaction force
    _tit = 'Reaction force at the beam tip, ' + str(pars.constraint)
    _name = 'cantBeam_reactionForce_compare_' + str(pars.constraint)
    _path = './examples/fem/models_QS/'
    plots_together(xs, ys, labels, x_label='u', y_label='f', _tit=_tit, _name=_name, _path=_path)
    
    ## Write parameters used
    _name_write = 'cantBeam_reactionForce_parameters_' + str(pars.constraint)
    write_attributes(pars, _name=_name_write, _path=_path)
    
    return plots_data1, plots_data2, plots_data3


if __name__ == "__main__":
    plots_data1, plots_data2, plots_data3 = run_3_simulations()
    
    df.parameters["form_compiler"]["representation"] = "uflacs" # back to default setup of dolfin
    
    