from feniQS.problem.QSM_GDM import *
from feniQS.structure.struct_slab2D import *

def revise_pars_gdm_slab2d(pars):
    pars.constraint = 'PLANE_STRAIN'
    pars.E_min = 40e3
    pars.E = 20e3 # E_total = 60e3
    pars.e0_min = 6e-4
    pars.e0 = 5e-4 # e0_total = 11e4
    pars.ef = 35e-4
    pars.c_min = 1.
    pars.c = 0.

    # pars.shF_degree_u = 2
    # pars.shF_degree_ebar = 1
    # pars.integ_degree = 3

def pp_qsm_gdm_slab2d(model, solve_options):
    pp0 = model.pps[0]
    pp0.plot_residual_at_free_DOFs('Residual-norm of free DOFs' \
                                , full_file_name=f"{model._path}residual_norm_free_DOFs.png")
    pp0.plot_reaction_forces(tits=[f"Reaction force at {rp}" for rp in solve_options.reaction_places])
    
if __name__ == "__main__":
    df.set_log_level(30)
    
    ## PARAMETERs
    pars_struct = ParsSlab2D()
    pars_struct.loading_level *= 0.5
    pars_gdm = GDMPars(pars0=pars_struct) # We merge pars_struct to gdm model pars (through pars0)
    revise_pars_gdm_slab2d(pars_gdm)
    
    ## MODEL (quasi static)
    model = get_QSM_GDM(pars_struct=pars_struct, cls_struct=Slab2D, pars_gdm=pars_gdm)
    
    ## SOLVE OPTIONs
    solver_options = get_fenicsSolverOptions() # regarding a single load-step
    solver_options['tol_abs'] = 1e-10
    solver_options['tol_rel'] = 1e-10
    # solver_options['type'] = 'snes'
    # solver_options['lin_sol'] = 'iterative'
    solve_options = QuasiStaticSolveOptions(solver_options) # regarding incremental solution
    n_ch = 10
    solve_options.checkpoints = [float(a) for a in (np.arange(1./n_ch, 1.0, 1./n_ch))]
    solve_options.t_end = model.pars.loading_t_end
    solve_options.dt = 0.01
    solve_options.reaction_places = ['left', 'right']
    
    ## SOLVE
    model.solve(solve_options) # 'build_solver' is included there.
    
    ## POST-PROCESS
    pp_qsm_gdm_slab2d(model, solve_options)
    
    df.set_log_level(20)
    