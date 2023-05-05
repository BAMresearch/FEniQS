from feniQS.problem.problem_elastic_homogenized import *

if __name__ == "__main__":
    df.set_log_level(20)
    
    ## MESH (of real structure)
    lx = 2.
    ly = 2.
    mesh = df.RectangleMesh(df.Point(0.,0.), df.Point(lx, ly), 5, 8)
    
    ## RVE's feature: RVE_volume & RVE_boundaries
    RVE_volume = estimate_RVE_volume(mesh)
    RVE_boundaries = get_RVE_boundaries(mesh)
    
    ## Homogenization parameters (based on elastic parameters)
    pars_elastic = ElasticPars()
    pars_elastic.E = 1000.
    pars_elastic.nu = 0.2
    pars_elastic.constraint = 'PLANE_STRESS'
    pars_hom = ParsHomogenization(pars0=pars_elastic)
    pars_hom.hom_type = 'Dirichlet'
    
    ## Homogenization problem
    hom_problem = build_homogenization_problem(mesh=mesh \
                                               , pars_hom=pars_hom \
                                               , RVE_volume=RVE_volume \
                                               , RVE_boundaries=RVE_boundaries)
    
    ## FEniCS solver
    so = get_fenicsSolverOptions()
    
    ## Solve problem
    hom_problem.build_solver(solver_options=so)
    hom_problem.run_hom()
    
    df.set_log_level(30)