import sys
if './' not in sys.path:
    sys.path.append('./')

from feniQS.problem.QSM_GDM import *
from examples.fem.structures.struct_kozicki2013 import *

pth_QSM_GDM_kozicki2013 = CollectPaths('QSM_GDM_kozicki2013.py')
pth_QSM_GDM_kozicki2013.add_script(pth_QSM_GDM)
pth_QSM_GDM_kozicki2013.add_script(pth_struct_kozicki2013)

def revise_pars_gdm_kozicki2013(pars):
    pars.constraint = 'PLANE_STRAIN'
    pars.E_min = 40e3
    pars.E = 20e3 # E_total = 60e3
    pars.e0_min = 6e-4
    pars.e0 = 5e-4 # e0_total = 11e4
    pars.ef = 35e-4
    pars.c_min = 40.
    pars.c = 0. # c_total = 40.
    
    # pars.shF_degree_u = 2
    # pars.shF_degree_ebar = 1
    # pars.integ_degree = 3

def pp_qsm_gdm_kozicki2013(model, solve_options):
    model.pps[0].plot_residual_at_free_DOFs('Residual-norm of free DOFs', full_file_name=model._path + 'residual_norm_free_DOFs.png')
    
    ## Plot CMOD (crack mouth opening displacement)
    p1 = np.array([0.5 * (model.pars.lx - model.pars.l_notch), 0.0])
    p2 = np.array([0.5 * (model.pars.lx + model.pars.l_notch), 0.0])
    p3 = np.array([0.5 * (model.pars.lx - model.pars.l_notch), model.pars.h_notch])
    p4 = np.array([0.5 * (model.pars.lx + model.pars.l_notch), model.pars.h_notch])
    ups = model.pps[0].eval_checked_u([p1, p2, p3, p4])
    cmods_bot = [0] + [us[1,0]-us[0,0] for us in ups]
    cmods_top = [0] + [us[3,0]-us[2,0] for us in ups]
    loaded_us_checked = [0] + evaluate_expression_of_t(model.time_varying_loadings['y_middle'], ts=model.pps[0].checkpoints)[1]
    fig = plt.figure()
    plt.plot(loaded_us_checked, cmods_bot, marker='<', label='bottom')
    plt.plot(loaded_us_checked, cmods_top, marker='>', label='top')
    plt.plot([-0.3, -0.3], [0.02, 0.04], marker='X')
    plt.text(-0.3, 0.02, f"(-0.3, 0.02)", ha='center', va='top')
    plt.text(-0.3, 0.04, f"(-0.3, 0.04)", ha='center', va='bottom')
    plt.title('Crack mouth opening displacement')
    plt.xlabel('Deflection')
    plt.ylabel('CMOD')
    plt.legend()
    plt.savefig(model._path + 'cmod_vs_us.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    if len(model.pps[0].reaction_dofs) > 0:
        ## versus time
        factor = -1e-3 # is only for getting positive values in kN in plots
        tits = []; _names = []
        for ii, rp in enumerate(solve_options.reaction_places):
            tits.append(f"Reaction force at {rp}")
            _names.append(model._path + f"reaction_force_{ii+1}.png")
        fss = model.pps[0].plot_reaction_forces(tits, factor=factor \
                                          , full_file_names=_names)
        fs = fss[0]
        ## versus deflection
        loaded_us = evaluate_expression_of_t(model.time_varying_loadings['y_middle'], ts=model.pps[0].ts)[1]
        fig = plt.figure()
        plt.plot(loaded_us, fs, marker='*')
        _ys = np.linspace(min(fs), max(fs), 100)
        plt.plot(len(_ys) * [-0.3], _ys, linestyle='dashed')
        plt.text(-0.3, 0, "$u=-0.3$", ha='left', va='bottom')
        plt.title('Load-displacement')
        plt.xlabel('Deflection (mm)')
        plt.ylabel('f ($kN$)')
        plt.savefig(model._path + 'load_displacement.png', bbox_inches='tight', dpi=300)
        
        ## versus CMOD
        fs_checked = [fs[i] for i, a in enumerate(loaded_us) if a in loaded_us_checked]
        cmods_ave = [0.5*(a+cmods_top[i]) for i,a in enumerate(cmods_bot)]
        fig = plt.figure()
        plt.plot(cmods_ave, fs_checked, marker='+', label='vs. average CMOD')
        plt.plot(cmods_top, fs_checked, marker='x', label='vs. top CMOD')
        _ys = np.linspace(min(fs_checked), max(fs_checked), 100)
        plt.plot(len(_ys) * [0.02], _ys, linestyle='dashed')
        plt.text(0.02, 0, "0.02", ha='left', va='bottom')
        plt.plot(len(_ys) * [0.04], _ys, linestyle='dashed')
        plt.text(0.04, 0, "0.04", ha='left', va='bottom')
        plt.title('Load vs. CMOD')
        plt.xlim((0, 0.2))
        plt.ylim((0, 2))
        plt.xlabel('CMOD')
        plt.ylabel('f ($kN$)')
        plt.legend()
        plt.savefig(model._path + 'fs_vs_cmod.png', bbox_inches='tight', dpi=300)
        
        plt.show()
    
    df.list_timings(df.TimingClear.keep, [df.TimingType.wall])
    return fss
    
if __name__ == "__main__":
    df.set_log_level(30)
    
    ## PARAMETERs
    pars_struct = ParsKozicki2013()
    # pars_struct.l_notch = 1. # No notch
    pars_struct.loading_level *= 2. # larger loading, causing more damage
    pars_gdm = GDMPars(pars0=pars_struct) # We merge pars_struct to gdm model pars (through pars0)
    revise_pars_gdm_kozicki2013(pars_gdm)
    pars_struct.resolutions['el_size_max'] = np.sqrt(pars_gdm.c_min) / 1.2
    pars_gdm.resolutions['el_size_max'] = pars_struct.resolutions['el_size_max']
    
    ## MODEL (quasi static)
    model = get_QSM_GDM(pars_struct=pars_struct, cls_struct=Kozicki2013, pars_gdm=pars_gdm)
    
    ## SOLVE OPTIONs
    solver_options = get_fenicsSolverOptions() # regarding a single load-step
    solver_options['tol_abs'] = 1e-10
    solver_options['tol_rel'] = 1e-10
    # solver_options['type'] = 'snes'
    # solver_options['lin_sol'] = 'iterative'
    solve_options = QuasiStaticSolveOptions(solver_options) # regarding incremental solution
    solve_options.checkpoints = [float(a) for a in (np.arange(1e-1, 1.0, 1e-1))] # this "float" is to be able to properly write "solve_options" into yamlDict file
    solve_options.t_end = model.pars.loading_t_end
    solve_options.dt = 0.01
    solve_options.reaction_places = ['y_middle', 'y_left', 'y_right']
    
    ## SOLVE
    model.solve(solve_options) # 'build_solver' is included there.
    # Write parameters to yaml files
    model.yamlDump_pars()
    solve_options.yamlDump_toDict(_path=model._path)
    
    ## POST-PROCESS
    fss = pp_qsm_gdm_kozicki2013(model, solve_options)
    
    df.set_log_level(20)
    