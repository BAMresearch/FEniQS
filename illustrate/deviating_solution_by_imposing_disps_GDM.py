from feniQS.structure.struct_slab2D_long import *
from feniQS.structure.struct_kozicki2013 import *
from feniQS.problem.QSM_GDM import *
import copy

if __name__=='__main__':
    df.set_log_level(30)

    _path = './illustrate/deviating_solution_by_imposing_disps_GDM/'

    ### STRUCTURE-1 ###
    pars_struct = ParsSlab2DLong()
    pars_struct.red_factor = 0.01 # imperfection of material at the middle
    struct = Slab2DLong(pars_struct, _path=f"{_path}structure/")
    reaction_places = ['left', 'right']

    ### STRUCTURE-2 ###
    # pars_struct = ParsKozicki2013()
    # struct = Kozicki2013(pars_struct, _path=f"{_path}structure/")
    # reaction_places = ['y_middle']

    ### GDM ###
    pars = GDMPars(pars0=pars_struct)
    pars.ef = 1e-3
    pars.c_min = (1.4 * struct.mesh.hmax()) ** 2
    pars.constraint = 'PLANE_STRAIN'
    qsm_path_direct = f"{_path}QSM_direct/"
    model = QSModelGDM(pars=pars, struct=struct, _path=qsm_path_direct)
    pp_K = PostProcessEvalFunc(model.fen.u_K_current)
    model.pps_default['pp_K'] = pp_K

    solver_options = get_fenicsSolverOptions()
    solver_options['tol_abs'] = 1e-11
    solver_options['tol_rel'] = 1e-13
    solve_options = QuasiStaticSolveOptions(solver_options)
    n_ch = 10
    solve_options.checkpoints = [float(a) for a in (np.arange(1. / n_ch, 1.0, 1. / n_ch))] # this "float" is to be able to properly write "solve_options" into yamlDict file
    solve_options.t_end = model.pars.loading_t_end
    if solve_options.t_end not in solve_options.checkpoints:
        solve_options.checkpoints.append(float(solve_options.t_end))
    solve_options.reaction_places = reaction_places

    model.solve(solve_options)
    K_checked_direct = copy.deepcopy(np.array(pp_K.checked))
    ts_direct = copy.deepcopy(model.pps[0].ts)
    Fs_direct = model.pps[0].plot_reaction_forces(reaction_places)
    f_max = np.max(abs(np.array(Fs_direct)))

    qsm_path_impose = f"{_path}QSM_impose/"
    model._path = qsm_path_impose
    res0 = QSModelGDM.solve_by_imposing_displacements_at_free_DOFs( \
        model_solved=model, solve_options=solve_options, _plot=True)
    K_checked_impose = copy.deepcopy(np.array(pp_K.checked))
    ts_impose = copy.deepcopy(model.pps[0].ts)
    Fs_impose = model.pps[0].plot_reaction_forces(reaction_places)

    plt.figure()
    for i, r0 in enumerate(res0):
        plt.plot(r0, linestyle='--', label=f"LS={i}, sum/f_max={(sum(r0) / f_max):.2e}")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('DOF'); plt.ylabel('Residual')
    plt.title(f"Residuals at free DOFs after imposing the solved displacements")
    plt.savefig(f"{qsm_path_impose}free_residuals.png", bbox_inches='tight', dpi=400)
    plt.show()

    for i,f_difect in enumerate(Fs_direct):
        f_impose = Fs_impose[i]
        rp = reaction_places[i]
        fig = plt.figure()
        plt.plot(ts_direct, f_difect, marker='.', label='Direct solution')
        plt.plot(ts_impose, f_impose, marker='o', fillstyle='none', label='Solved by imposed displacements')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"Reaction force at {rp}")
        plt.xlabel('t'); plt.ylabel('F')
        plt.savefig(f"{_path}compare_reaction_force_{rp}.png", bbox_inches='tight', dpi=400)
        plt.show()

    rel_err = 0.03
    path_GPs = f"{_path}compare_history_variable/"
    make_path(path_GPs)
    bad_GPs_ids = []
    for i, K_direct in enumerate(K_checked_direct):
        K_impose = K_checked_impose[i,:]
        for j, K_d in enumerate(K_direct):
            ref = K_d if K_d>1e-6 else 1.
            err = abs(K_d - K_impose[j]) / ref
            if err > rel_err:
                if j not in bad_GPs_ids:
                    bad_GPs_ids.append(j)
        plt.figure()
        plt.plot(K_direct, linestyle='', marker='.', label=f"Direct")
        plt.plot(K_impose, linestyle='', marker='o', fillstyle='none', label=f"Imposed")
        plt.xlabel(f"Gauss point ID"); plt.ylabel(f"Kappa")
        plt.title(f"Load-step = {i}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"{path_GPs}compare_history_LS_{i}.png", bbox_inches='tight', dpi=400)
        plt.show()
    
    bad_GPs = np.array([model.fen.i_K.tabulate_dof_coordinates()[ii, :] for ii in bad_GPs_ids])
    fig = plt.plot()
    df.plot(model.fen.mesh, color='gray', linewidth=min(1., model.struct.mesh.rmin()/1.))
    if len(bad_GPs)>0:
        plt.plot(bad_GPs[:,0], bad_GPs[:,1], marker='*', label='Inaccurate GPs', linestyle='')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Inaccurate GPs with relative error of {rel_err}")
    plt.savefig(f"{path_GPs}inaccurate_GPs.png", bbox_inches='tight', dpi=500)
    plt.show()

    df.set_log_level(20)
