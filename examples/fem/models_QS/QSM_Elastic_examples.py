from feniQS.problem.QSM_Elastic import *
from feniQS.structure.struct_box_compressed import *
from feniQS.structure.struct_bcc_compressed import *

if __name__ == "__main__":
    df.set_log_level(20)
    
# =============================================================================
#     STRUCTURE
# =============================================================================
    ## Example (1): BOX COMPRESSED ##
    # cls_struct = BoxCompressed
    # pars_struct = ParsBoxCompressed() # Initialize model parameters
    # res                 = 5
    # pars_struct.res_x   = res
    # pars_struct.res_y   = res
    # pars_struct.res_z   = res
    
    ## Example (2): BCC COMPRESSED ##
    cls_struct = BccCompressed
    pars_struct = ParsBccCompressed() # Initialize model parameters
    ## CASE-1: parametric nominal mesh
    pars_struct.n_rve       = 2
    pars_struct.l_rve       = 3.333333
    pars_struct.r_strut     = 0.5
    pars_struct.add_plates  = True
    pars_struct.l_cell      = 0.4
    ## CASE-2: mesh from a file
    # pars_struct.link_mesh_xdmf = './examples/fem/mesh_files/bcc/bcc_1/220214_test_A.xdmf'
    
# =============================================================================
#     QUASI-STATIC MODEL (ELASTIC)
# =============================================================================
    pars_elastic = ElasticPars(pars0=pars_struct) # We merge pars_struct to elastic model pars (through pars0)
    pars_elastic.constraint = '3D'
    
    model = get_QSM_Elastic(pars_struct     = pars_struct,
                            cls_struct      = cls_struct,
                            pars_elastic    = pars_elastic)
    
# =============================================================================
#     SOLVE & PP (post-processing)
# =============================================================================
    ## (SOLVE & SOLVER) OPTIONs
    solver_options = get_fenicsSolverOptions(case='linear', lin_sol='iterative')
    solver_options['lin_sol_options']['method'] = 'cg'
    solver_options['lin_sol_options']['precon'] = 'default'
    solver_options['lin_sol_options']['tol_abs'] = 1e-10
    solver_options['lin_sol_options']['tol_rel'] = 1e-10
    
    solve_options = QuasiStaticSolveOptions(solver_options=solver_options)
    n_ch = 2
    solve_options.checkpoints = [float(a) for a in np.linspace(solve_options.t_end/n_ch, solve_options.t_end, n_ch)]
        # this "float" is to be able to properly write "solve_options" into yamlDict file
    solve_options.dt = solve_options.t_end / n_ch
    solve_options.reaction_places   = ['z_top']
    
    ## SOLVE
    model.solve(solve_options)
    
    ## POST-PROCESS
    pp0 = model.pps[0]
    pp0.plot_reaction_forces(tits=[f"Reaction force at {rp}" for rp in solve_options.reaction_places])
    
    df.set_log_level(20)
