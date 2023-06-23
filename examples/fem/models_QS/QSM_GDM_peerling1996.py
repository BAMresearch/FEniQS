from feniQS.problem.QSM_GDM import *
from feniQS.structure.struct_peerling1996 import *

pth_QSM_GDM_peerling1996 = CollectPaths('./examples/fem/models_QS/QSM_GDM_peerling1996.py')
pth_QSM_GDM_peerling1996.add_script(pth_QSM_GDM)
pth_QSM_GDM_peerling1996.add_script(pth_struct_peerling1996)

def revise_pars_gdm_peerling1996(pars):
    ##### Each of the following factors results in the same load-disp. curve (energy dissipation).
    f_ef = 1.; f_c = 1. ** 2
    # f_ef = 0.7; f_c = 1.61 ** 2
    # f_ef = 0.582; f_c = 2. ** 2
    # f_ef = 0.5; f_c = 2.37 ** 2
    # f_ef = 0.3; f_c = 4.03 ** 2
    # f_ef = 0.4; f_c = 3.00 ** 2
    # f_ef = 0.8; f_c = 1.36 ** 2
    #######
    pars.constraint = 'UNIAXIAL'
    pars.E_min = 5e3
    pars.E = 25e3 # E_total = 30e3
    pars.e0_min = 0.5e-4
    pars.e0 = 0.5e-4 # e0_total = 1e-4
    pars.ef_min = 0.
    pars.ef = f_ef * 10e-4 # ef_total = ef
    
    pars.c_min = f_c * (pars.geo_scale ** 2)
    pars.c = f_c * (pars.geo_scale ** 2) # c_total = f_c * 2 * (pars.geo_scale**2)
    
    pars.damage_law = 'exp'
    # pars.damage_law = 'perfect'

def integrate_load_disp_plot(model):
    pp0 = model.pps[0]
    if model.pars.loading_control=='u':
        fs_side = [np.sum(ff) for ff in pp0.reaction_forces[1]]
        us = evaluate_expression_of_t(model.struct.u_right, ts=pp0.ts)[1]
    elif model.pars.loading_control=='f':
        fs_side = evaluate_expression_of_t(model.struct.f_right, ts=pp0.checkpoints)[1]
        us = [v[0] for v in pp0.eval_checked_u([[model.pars.L]])]
    else:
        raise ValueError(f"Loading control of the structure {model.struct._name} is not recognized.")
    plt.figure()
    plt.plot(us, fs_side, marker='.', linestyle='--')
    plt.title('Load-displacement')
    plt.xlabel('U')
    plt.ylabel('F')
    plt.show()
    w = area_under_XY_data(us, fs_side)
    return w

def study_energy_peerling1996(model):
    E = model.pars.E.values()[0] + model.pars.E_min
    e0 = model.pars.e0.values()[0] + model.pars.e0_min
    ef = model.pars.ef.values()[0] + model.pars.ef_min
    c = model.pars.c.values()[0] + model.pars.c_min
    l = c ** 0.5
    w_per_volume = model.fen.mat.gK.maximum_dissipation(E=E, e0=e0, ef=ef)
    w = integrate_load_disp_plot(model)
    L = w / w_per_volume
    print(f"\n\t\tTotal energy dissipated = {w:.3e} .\n\t\t\tThis is also per area, since cross area is 1 .")
    print(f"\n\t\tMaximum energy dissipation (per volume) from damage law = {w_per_volume:.3e} .")
    print(f"\n\t\tL = Rough estimation of damaged volume = {L:.3e} .\n\t\t\tThis is also per length, since cross area is 1 .")
    print(f"\n\t\tDamage parameter 'l' = sqrt(c) = {l:.3e}. -->  L / l = {(L/l):.3e}")

def add_pps_for_animations_peerling1996(model, solve_options):
    pp_base = model.pps[0]
    
    u_dofs = model.fen.i_mix.sub(0).dofmap().dofs()
    e_dofs = model.fen.i_mix.sub(1).dofmap().dofs()
    coors = model.fen.i_mix.tabulate_dof_coordinates()
    coors_u = coors[u_dofs]
    coors_e = coors[e_dofs]
    u_dofs_sorted = [x for _,x in sorted(zip(coors_u, u_dofs))] # sorted based on coordinates
    e_dofs_sorted = [x for _,x in sorted(zip(coors_e, e_dofs))] # sorted based on coordinates
    
    model.pp_u = PostProcessEvalFunc(func=model.fen.u_mix, dofs=u_dofs_sorted, checkpoints=solve_options.checkpoints)
    model.pps.append(model.pp_u)
    model.pp_K = PostProcessEvalFunc(func=pp_base.K, dofs=pp_base._q_space.dofmap().dofs(), checkpoints=solve_options.checkpoints)
    model.pps.append(model.pp_K)
    model.pp_ebar = PostProcessEvalFunc(func=model.fen.u_mix, dofs=e_dofs_sorted, checkpoints=solve_options.checkpoints)
    model.pps.append(model.pp_ebar)
    model.pp_D = PostProcessEvalFunc(func=pp_base.D, dofs=pp_base._q_space.dofmap().dofs(), checkpoints=solve_options.checkpoints)
    model.pps.append(model.pp_D)
    
def pp_qsm_gdm_peerling1996(model, _animate=True):
    plt.figure()
    df.plot(model.fen.get_uu())
    sz = 18
    plt.title('Final displacement along the bar', fontsize=sz)
    plt.xlabel('x', fontsize=sz)
    plt.ylabel('u_x', fontsize=sz)
    plt.xticks(fontsize=sz);
    plt.yticks(fontsize=sz);
    
    tits = ['Reaction force 1', 'Reaction force 2']
    _names = [model._path+'reaction_force_1.pdf']
    _names.append(model._path+'reaction_force_2.pdf')
    model.pps[0].plot_reaction_forces(tits, full_file_names=_names, sz=sz)
    
    model.pps[0].plot_residual_at_free_DOFs('Residual-norm of free DOFs', full_file_name=model._path+'residual_norm_free_DOFs.pdf')
    
    if _animate:
        dpi = 100
        _l = len(model.pp_u.all[0])
        animate_list_of_2d_curves(x_fix=np.linspace(0, model.pars.L, _l), ys_list=model.pp_u.all \
                                  , _delay=200, xl='Bar length', yl='Displacement', _tit='Displacements evolution' \
                                    , _marker='.', _path=model._path, _name='Us_evolution', dpi=dpi)
        _l = len(model.pp_K.all[0])
        animate_list_of_2d_curves(x_fix=np.linspace(0, model.pars.L, _l), ys_list=model.pp_K.all \
                                  , _delay=200, xl='Bar length', yl='$\kappa$', _tit='History variable evolution' \
                                      , _marker='.', _path=model._path, _name='Kappas_evolution', dpi=dpi)
        _l = len(model.pp_ebar.all[0])
        animate_list_of_2d_curves(x_fix=np.linspace(0, model.pars.L, _l), ys_list=model.pp_ebar.all \
                                  , _delay=200, xl='Bar length', yl='Nonlocal equivalent strain', _tit='E_bar evolution' \
                                      , _marker='.', _path=model._path, _name='E_bar_evolution', dpi=dpi)
        _l = len(model.pp_D.all[0])
        animate_list_of_2d_curves(x_fix=np.linspace(0, model.pars.L, _l), ys_list=model.pp_D.all \
                                  , _delay=200, xl='Bar length', yl='$D$', _tit='Damage evolution' \
                                      , _marker='.', _path=model._path, _name='Damage_evolution', dpi=dpi)
        plt.show()
    
def several_imperfections():
    pars_struct = ParsPeerling()
    pars_struct._res = 200
    
    solve_options = QuasiStaticSolveOptions(solver_options=get_fenicsSolverOptions())
    solve_options.t_end = pars_struct.loading_t_end
    solve_options._plot = False
    solve_options.reaction_places = ['right']
    solve_options.solver_type = 'snes'
    
    models = []
    xs_defect = []
    for x_c in pars_struct.L * np.array([0.2, 0.5, 0.8]):
        pars_struct.x_from = x_c - 0.1 * pars_struct.L
        pars_struct.x_to = x_c + 0.1 * pars_struct.L
        pars = GDMPars(pars0=pars_struct) # We merge pars_struct to gdm model pars (through pars0)
        revise_pars_gdm_peerling1996(pars)
        model = get_QSM_GDM(pars_struct=pars_struct, cls_struct=Peerling1996, pars_gdm=pars)
        xs_defect.append(0.5 * (pars_struct.x_from + pars_struct.x_to))
        models.append(model)
        model.solve(solve_options)
    sz = 16
    plt.figure()
    for i, m in enumerate(models):
        df.plot(m.fen.get_uu(), label="$x_{defected}$ = " + str(xs_defect[i]))
        # df.plot(m.fen.get_uu(), label="$x_{defected,"+str(i)+"}$")
    plt.title('Final displacement along the bar', fontsize=sz)
    plt.xlabel('$x$', fontsize=sz)
    plt.ylabel('$u_x$', fontsize=sz)
    plt.xticks(fontsize=sz);
    plt.yticks(fontsize=sz);
    plt.legend(fontsize=0.8*sz, loc='lower right')
    plt.savefig(models[0]._path+'several_us_bar.png', bbox_inches='tight', dpi=300)
    plt.show()
    return models

def several_interpolateDegU(degs=[1, 2, 3]):
    pars_struct = ParsPeerling()
    pars_struct._res = 200
    
    solve_options = QuasiStaticSolveOptions(solver_options=get_fenicsSolverOptions())
    solve_options.t_end = pars_struct.loading_t_end
    solve_options._plot = False
    solve_options.reaction_places = ['right']
    solve_options.checkpoints = [float(a) for a in (np.arange(1e-1, 1.0, 1e-1))]
    solve_options.solver_type = 'snes'
    
    pars = GDMPars(pars0=pars_struct) # We merge pars_struct to gdm model pars (through pars0)
    revise_pars_gdm_peerling1996(pars)
    
    models = []
    Us = []
    for deg in degs:
        pars.shF_degree_u = deg
        pars.integ_degree = deg + 1
        model = get_QSM_GDM(pars_struct=pars_struct, cls_struct=Peerling1996, pars_gdm=pars)
        points = model.fen.mesh.coordinates()
        models.append(model)
        model.solve(solve_options)
        Us.append(model.pps[0].eval_checked_u(points))
    sz = 16
    num_ch = len(Us[0])
    for i in range(num_ch):
        us = [u[i] for u in Us]
        plt.figure()
        for j, u in enumerate(us):
            deg = degs[j]
            plt.plot(u, label=f"Interp. deg. = {deg}")
        plt.title(f"Displacement at checkpoint {i}" , fontsize=sz)
        plt.xlabel('$x$', fontsize=sz)
        plt.ylabel('$u_x$', fontsize=sz)
        plt.xticks(fontsize=sz);
        plt.yticks(fontsize=sz);
        plt.legend(fontsize=0.8*sz, loc='lower right')
        plt.savefig(models[0]._path+f"several_interpolate_degs_ch_{i}.png", bbox_inches='tight', dpi=300)
    plt.show()
    return models
    
if __name__ == "__main__":
    df.set_log_level(30)
    
    ## PARAMETERs
    pars_struct = ParsPeerling()
    pars_struct._res = 200

    pars_struct.loading_level /= 4
    pars_gdm = GDMPars(pars0=pars_struct) # We merge pars_struct to gdm model pars (through pars0)
    revise_pars_gdm_peerling1996(pars_gdm)
    
    ## MODEL (quasi static)
    model = get_QSM_GDM(pars_struct=pars_struct, cls_struct=Peerling1996, pars_gdm=pars_gdm)
    
    ## SOLVE OPTIONs
    solve_options = QuasiStaticSolveOptions(solver_options=get_fenicsSolverOptions())
    solve_options.checkpoints = [float(a) for a in (np.arange(2e-1, 1.0, 2e-1))] # this "float" is to be able to properly write "solve_options" into yamlDict file
    solve_options.t_end = model.pars.loading_t_end
    solve_options.dt = 0.01
    solve_options.reaction_places = ['left', 'right']
    
    ## ADD PPs for animations
    model.set_pps(solve_options, DG_degree=0) # We set_pp before calling the solver, since we need base_pp=model.pps[0] in calling add_pps_for_animations_peerling1996
    add_pps_for_animations_peerling1996(model, solve_options)
    
    ## SOLVE
    model.build_solver(solve_options) # Crucial to build solver, since we have BCs.
    model.solve(solve_options, _reset_pps=False) # We already set pps.
    # Write parameters to yaml files
    model.yamlDump_pars()
    solve_options.yamlDump_toDict(_path=model._path)
    
    ## POST-PROCESS
    pp_qsm_gdm_peerling1996(model, _animate=False)
    
    ## STUDY ENERGY
    study_energy_peerling1996(model)
    
    ## STUDY DEFFECT POSITION
    models = several_imperfections()
    
    ## STUDY INTERPOLATION DEGREE
    models2 = several_interpolateDegU()
    
    df.set_log_level(20)
    