from feniQS.structure.struct_cantilever_beam import *
from feniQS.problem.QSM_Plasticity import *

def revise_pars_plasticity_cantileverBeam2d(pars):
    pars.shF_degree_u = 3
    pars.integ_degree = 4

    pars.constraint = 'PLANE_STRESS'
    pars.E = 1000.
    pars.nu = 0.3
    pars.yield_surf = {'type': 'von-mises'}
    pars.yield_surf['pars'] = {'sig0': 12.0}

    ### ISOTROPIC Hardening modulus
        # (1)
    Et = pars.E / 100.0
    H = 15 * pars.E * Et / (pars.E - Et)
        # (2) No hardening
    # H = 0.0

    pars.hardening_isotropic['modulus'] = H
    pars.hardening_isotropic['law'] = 'linear'
    pars.hardening_isotropic['hypothesis'] = 'unit' # or 'plastic-work'

def pp_plasticity_cantileverBeam2d(model, solve_options, lz=1.0, sz=14, _format='.png'):
    pp0 = model.pps[0]
    fss = pp0.plot_reaction_forces(tits=[f"Reaction force at {rp}" for rp in solve_options.reaction_places])
    
    plt.figure()
    _, u_sol = evaluate_expression_of_t(model.struct.u_y_tip, ts=pp0.ts)
    f_sol = abs(np.array([sum(fs) for fs in pp0.reaction_forces[0]]))
    plt.plot(u_sol, lz * f_sol, marker='.', label='FEniCS: Plasticity')
    E_total = model.pars.E + model.pars.E_min
    K_analytical = lz * (model.pars.ly ** 3) * E_total / 4. / (model.pars.lx ** 3)
    u_analytical = [u for u in u_sol if (K_analytical*u<=max(f_sol) and K_analytical*u>=min(f_sol))]
    f_analytical = K_analytical * np.array(u_analytical)
    plt.plot(u_analytical, f_analytical, label=f"Equivalent elastic stiffness = {K_analytical:.1e}")
    if (model.pars.hardening_isotropic['modulus'] == 0 \
        and model.pars.loading.level == 0.65 \
            and model.pars.loading.case=='ramp'):
        _u, _f = read_ref_sol_MATLAB()
        plt.plot(_u, abs(lz * _f), marker='', label='MATLAB')
    _tit = 'Reaction force at the beam tip'
    plt.title(_tit, fontsize=sz)
    plt.xlabel('u', fontsize=sz)
    plt.ylabel('abs(f)', fontsize=sz)
    plt.xticks(fontsize=sz)
    plt.yticks(fontsize=sz)
    plt.legend()
    plt.savefig(f"{model._path}{model._name}_reaction_force_compare{_format}" \
                , bbox_inches='tight', dpi=300)
    plt.show()

    return [[u_analytical, u_sol], [f_analytical, f_sol], ['Equivalent elastic stiffness', 'FEniCS: Plasticity']]

def read_ref_sol_MATLAB(_file = './examples/fem/models_QS/cantBeamPlasticRefSol2d_MATLAB/Results.mat'):
    from scipy.io import loadmat
    _data = loadmat(_file)
    u = _data['results'][0][0][0][0]
    f = _data['results'][0][0][1][0]
    return u, -f

if __name__ == "__main__":
    df.set_log_level(30)
    
    pars_struct = ParsCantileverBeam2D()
    pars_struct.lx = 6.0
    pars_struct.ly = 0.5
    pars_struct.res_x = 36
    pars_struct.res_y = 3
    pars_struct.loading_level = 4. * pars_struct.ly

    pars_plastic = PlasticityPars(pars0=pars_struct)
    revise_pars_plasticity_cantileverBeam2d(pars_plastic)

    model_plastic = get_QSM_Plasticity(pars_struct, CantileverBeam2D, pars_plastic)

    ## SOLVE OPTIONs
    solver_options = get_fenicsSolverOptions(lin_sol='direct') # regarding a single load-step
    solve_options = QuasiStaticSolveOptions(solver_options) # regarding incremental solution
    solve_options.t_end = model_plastic.pars.loading_t_end
    n_ch = 10
    solve_options.checkpoints = [float(a) for a in np.linspace(solve_options.t_end/n_ch, solve_options.t_end, n_ch)]
    solve_options.dt = 0.01
    solve_options.dt_max = 0.05
    solve_options.reaction_places = ['left_y', 'right_y']
    
    ## SOLVE
    model_plastic.solve(solve_options) # 'build_solver' is included there.

    ## POST-PROCESS
    pp_outputs = pp_plasticity_cantileverBeam2d(model_plastic, solve_options)

    df.set_log_level(20)
