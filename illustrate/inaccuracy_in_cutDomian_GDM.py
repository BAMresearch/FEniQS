"""
Created on Fri Feb 18 15:21:57 2022
@author: ajafari
This Example illustrates:
    we get inaccurate nonlocal-field if we impose solution of a full-domain to sub-domain.
    The reason is that the Neuman BC of the second problem (on the sub-domain)
        is fullfilled over other boundaries (of the cutted domain) than original problem's.
"""

# from problems.QSM_GDM_kozicki2013 import *
from feniQS.structure.struct_bend3point2d_cutMiddle import *
from feniQS.structure.struct_kozicki2013 import *
from feniQS.problem.QSM_GDM import *

def revise_pars_gdm_kozicki2013(pars):
    pars.constraint = 'PLANE_STRAIN'
    pars.E_min = 40e3
    pars.E = 20e3 # E_total = 60e3
    pars.e0_min = 6e-4
    pars.e0 = 5e-4 # e0_total = 11e4
    pars.ef = 35e-4
    pars.c_min = 40.
    pars.c = 0. # c_total = 40.

def main(_range=[0.2, 0.8]):
    ## Generate and solve the full model (over full domain
    pars_struct = ParsKozicki2013()
    
    pars = GDMPars(pars0=pars_struct) # We merge pars_struct to gdm model pars (through pars0)
    revise_pars_gdm_kozicki2013(pars)
    pars_struct.resolutions['el_size_max'] = np.sqrt(pars.c_min) / 1.2
    pars.resolutions['el_size_max'] = pars_struct.resolutions['el_size_max']

    model = get_QSM_GDM(pars_struct=pars_struct, cls_struct=Kozicki2013, pars_gdm=pars)
    solve_options = QuasiStaticSolveOptions(solver_options=get_fenicsSolverOptions())
    solve_options.checkpoints = [float(a) for a in (np.arange(1e-1, 1.0, 1e-1))] + [1.0]
    solve_options.t_end = model.pars.loading_t_end
    model.solve(solve_options)
    
    ## Generate cutted structure (at the middle)
    mesh = model.fen.mesh
    Cs = mesh.coordinates()
    x1, x2 = min(Cs[:,0]), max(Cs[:,0])
    x_min = x1 + _range[0] * (x2-x1)
    x_max = x1 + _range[1] * (x2-x1)
    
    # WAY-1 (directly using developed class 'Bend3Point2D_cutMiddle')
    struct_sub = Bend3Point2D_cutMiddle(parent_struct=model.struct, intervals_or_nodes=[[x_min, x_max]], load_at_middle=False)
    mesh_sub = struct_sub.mesh
    
    # WAY-2 (using sub_mesh manually)
    # cs = []
    # for c in Cs:
    #     if c[0]>x_min and c[0]<x_max:
    #         cs.append(c)
    # cs = np.array(cs)
    # mesh_sub, alone_nodes = subMesh_over_nodes(mesh, cs)
    # struct_sub = StructureFromMesh(mesh_sub)
    
    ## Build sub-model (GDM model based on cutted structure)
    model_sub = QSModelGDM(pars, struct_sub, _name=model._name+'_sub')
    
    ## Get the displacements of the solution of the full model over all nodes of sub-mesh
    imposed_points = mesh_sub.coordinates()
    imposed_us = model.pps[0].eval_checked_u(imposed_points)
    
    ## Solve the sub-model by imposing displacements of full model
    many_bc = ManyBCs(V=model_sub.fen.get_i_full(), v_bc=model_sub.fen.get_iu(), points=imposed_points, sub_ids=[0])
    initiated_bcs = [many_bc.bc]
    bcs_assigner = many_bc.assign
    model_sub.revise_BCs(remove=True, new_BCs=initiated_bcs, _as='inhom')
    evolver = ValuesEvolverInTime(bcs_assigner, imposed_us, solve_options.checkpoints)
    model_sub.prep_methods = [evolver]
    solve_options.reaction_places = []
    model_sub.build_solver(solve_options) # After a change of BCs, rebuilding the solver is essential.
    model_sub.solve(solve_options)
    
    plt.figure()
    df.plot(mesh, color='gray')
    df.plot(mesh_sub, color='Blue')
    plt.title('Full and cut meshes')
    plt.savefig('./illustrate/full_and_sub_domains.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('DONE !\n\nSee the results in paraview.\nThe nonlocal fields are not identical, although the displacements are exactly the same.')
    print(f"\nResults on:\n{model._path}\nand\n{model_sub._path}")
    
    return model, model_sub

if __name__=="__main__":
    df.set_log_level(30)
    
    model, model_sub = main()
    
    df.set_log_level(20)
