import sys
if './' not in sys.path:
    sys.path.append('./')

from feniQS.problem.problem_elastic_homogenized import *

if __name__ == "__main__":
    df.set_log_level(20)
    
    # MESH (of real structure)
    lx = 2.
    ly = 2.
    mesh = df.RectangleMesh(df.Point(0.,0.), df.Point(lx, ly), 5, 8)
    
    # RVE's feature: RVE_boundaries & RVE_volume
    tol = mesh.rmin() / 1000.
    def RVE_boundaries(x, on_boundary):
        return df.near(x[0], 0., tol) \
            or df.near(x[1], 0., tol) \
                or df.near(x[0], lx, tol) \
                    or df.near(x[1], ly, tol)
    RVE_volume = lx * ly
    
    # Homogenization problem
    pars_elastic = ElasticPars()
    pars_elastic.E = 1000.
    pars_elastic.nu = 0.2
    pars_elastic.constraint = 'PLANE_STRESS'
    pars_hom = ParsHomogenization(pars0=pars_elastic)
    pars_hom.RVE_volume = RVE_volume
    pars_hom.hom_type = 'Dirichlet'
    hom_problem = build_homogenization_problem(mesh, pars_hom, RVE_boundaries=RVE_boundaries)
    
    # Solve
    so = get_fenicsSolverOptions()
    hom_problem.build_solver(solver_options=so)
    hom_problem.run_hom()
    
    df.set_log_level(30)