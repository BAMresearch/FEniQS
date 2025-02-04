from feniQS.problem.QSM_Elastic import *
from feniQS.structure.struct_Lshape2D import *

def revise_pars_elastic_lshape2d(pars):
    pars.constraint = 'PLANE_STRAIN'
    pars.E_min = 0.
    pars.E = 50e3 # E_total = 60e3

    # pars.shF_degree_u = 2
    # pars.integ_degree = 2

def pp_qsm_elastic_lshape2d(model, solve_options):
    pp0 = model.pps[0]
    pp0.plot_residual_at_free_DOFs('Residual-norm of free DOFs' \
                                , full_file_name=f"{model._path}residual_norm_free_DOFs.png")
    pp0.plot_reaction_forces(tits=[f"Reaction force at {rp}" for rp in solve_options.reaction_places])

if __name__ == "__main__":
    df.set_log_level(30)
    
    ## PARAMETERs
    pars_struct = ParsLshape2D()
    pars_struct.ly = 10 * pars_struct.lx
    pars_struct.wx = 0.6 * pars_struct.lx
    pars_struct.wy = 0.05 * pars_struct.ly
    pars_struct.resolutions['res_x'] = 6
    pars_struct.resolutions['res_y'] = 10 * pars_struct.resolutions['res_x']
    embedded_nodes = ParsLshape2D.get_regular_grid_points(pars_struct \
                                    , finement=1.0, portion=1.0)
    ParsLshape2D.set_embedded_nodes(pars_struct, embedded_nodes)
    rps_side = [] # reaction_places at the side edge
        # Different options for loading_control:
    pars_struct.loading_control = 'u_pull'; rps_side.append('side_x'); pars_struct.loading_level = 0.01 * pars_struct.ly
    # pars_struct.loading_control = 'u_shear'; rps_side.append('side_y'); pars_struct.loading_level = 0.01 * pars_struct.ly
    # pars_struct.loading_control = 'f_pull'; pars_struct.loading_level *= 100.
    # pars_struct.loading_control = 'f_shear'; pars_struct.loading_level *= 100.

    pars_elastic = ElasticPars(pars0=pars_struct) # We merge pars_struct to gdm model pars (through pars0)
    revise_pars_elastic_lshape2d(pars_elastic)
    
    ## MODEL (quasi static)
    model = get_QSM_Elastic(pars_struct=pars_struct, cls_struct=Lshape2D, pars_elastic=pars_elastic)
    
    ## SOLVE OPTIONs
    solver_options = get_fenicsSolverOptions(case='linear', lin_sol='default') # regarding a single load-step
    solve_options = QuasiStaticSolveOptions(solver_options) # regarding incremental solution
    n_ch = 10
    solve_options.checkpoints = [float(a) for a in (np.arange(1./n_ch, 1.0, 1./n_ch))]
    solve_options.t_end = model.pars.loading_t_end
    solve_options.dt = 0.01
    solve_options.reaction_places = rps_side + ['bot_x', 'bot_y']
    
    ## SOLVE
    model.solve(solve_options) # 'build_solver' is included there.
    
    ## POST-PROCESS
    pp_qsm_elastic_lshape2d(model, solve_options)
    
    df.set_log_level(20)