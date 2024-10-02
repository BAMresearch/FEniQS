from feniQS.structure.helper_mesh_gmsh_meshio import extrude_triangled_mesh_by_meshio
from feniQS.problem.problem import *
from feniQS.fenics_helpers.fenics_expressions import *

class MyFenicsConfig:
    def __init__(self):
        self.el_family      = 'CG'
        self.shF_degree_u   = 1
        self.integ_degree   = 2

if __name__=='__main__':

    ##### SURFACE MESH (input) #####
    ## (1) ##
    ff_mesh_surface = "./illustrate/files/disk.xdmf"
    ## (2) ##
    # ff_mesh_surface = './illustrate/files/mesh_2d.xdmf'
    # mesh_surface = df.RectangleMesh(df.Point(0.,0.), df.Point(1.,1.), 3, 3)
    # plt.figure()
    # df.plot(mesh_surface)
    # with df.XDMFFile(ff_mesh_surface) as ff:
    #     ff.write(mesh_surface)

    ##### EXTRUSION SETUP #####
    dz = lambda x: (np.array([0.,0.,50.]), np.array([0.,0.,-20])); res=[50,30]
    # dz=(50,-20); res=[50,30]
    # dz=50; res=50
    
    ##### EXTRUSION #####
    ff_mesh_extruded = './illustrate/files/mesh_extruded.xdmf'
    extrude_triangled_mesh_by_meshio(ff_mesh_surface=ff_mesh_surface \
                                    , ff_mesh_extruded=ff_mesh_extruded \
                                    , dz=dz, res=res)
    
    ##### MESH OBJECTs #####
    mesh_surface = df.Mesh()
    with df.XDMFFile(ff_mesh_surface) as ff:
        ff.read(mesh_surface)
    mesh_extruded = df.Mesh()
    with df.XDMFFile(ff_mesh_extruded) as ff:
        ff.read(mesh_extruded)
    
    ##### MESSAGEs #####
    _msg = f"The extruded mesh has {mesh_extruded.num_vertices()} nodes"
    _msg += f" and {mesh_extruded.num_cells()} elements."
    _msg += f"\nIt is stored in:\n\t'{ff_mesh_extruded}' ."
    print(_msg)
    
    ##### SIMULATION (to verify healthiness of mesh) #####
    _path = './illustrate/files/mesh_extruded_simulation/'
    make_path(_path)
    ff_xdmf = f"{_path}mesh_extruded_simulated.xdmf"
    if os.path.isfile(ff_xdmf):
        os.remove(ff_xdmf)
        os.remove(ff_xdmf.replace('.xdmf', '.h5'))
    xdmf = df.XDMFFile(ff_xdmf)
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.parameters["flush_output"] = True
    mat = ElasticConstitutive(E=1000., nu=0.2, constraint='3D')
    fen_config = MyFenicsConfig()
    fen = FenicsElastic(mat=mat, mesh=mesh_extruded, fen_config=fen_config)
    fen.build_variational_functionals()
    cs = fen.mesh.coordinates()
    z1, z2 = min(cs[:,2]), max(cs[:,2])
    tol = fen.mesh.rmin() / 1000.
    def bottom(x, on_boundary):
        return df.between(x[2], (z1 - tol, z1 + tol))
    def top(x, on_boundary):
        return df.between(x[2], (z2 - tol, z2 + tol))
    l_z = abs(z2 - z1); u_z = 0.02 * l_z
    u_top = ramp_expression(f_max=u_z, _tit='Ramp loading' \
                            , _name='ramp_loading', _path=_path)
    bc_bot, bc_bot_dofs = boundary_condition(i=fen.get_iu() \
                        , u=df.Constant((0., 0., 0.)), x=bottom)
    bc_top, bc_top_dofs = boundary_condition(i=fen.get_iu().sub(2) \
                        , u=u_top, x=top)
    fen.bcs_DR = [bc_bot]
    fen.bcs_DR_dofs = bc_bot_dofs
    fen.bcs_DR_inhom = [bc_top]
    fen.bcs_DR_inhom_dofs = bc_top_dofs
    solver_options = get_fenicsSolverOptions(case='linear', lin_sol='iterative')
    solver_options['lin_sol_options']['method'] = 'cg'
    solver_options['lin_sol_options']['precon'] = 'default'
    solver_options['lin_sol_options']['tol_abs'] = 1e-10
    solver_options['lin_sol_options']['tol_rel'] = 1e-10
    fen.build_solver(solver_options=solver_options, time_varying_loadings=[u_top])
    xdmf.write(fen.get_uu(), 0.)
    fen.solve(t=1.)
    xdmf.write(fen.get_uu(), 1.)
    xdmf.close()
    print(f"Elastic simulation finished.")
    ## Check
    vol = sum(get_element_volumes(fen.mesh))
    area_size_indicator = np.sqrt(vol / l_z)
    if l_z / area_size_indicator > 30.: # We can assume it is a long bar
        f_z = sum(df.assemble(fen.get_F_and_u()[0])[fen.bcs_DR_inhom_dofs])
        # The formula for tension of a long bar: Deflection = F * (l^2) / (Volume * E_modulus)
        u_z_analytic = f_z * (l_z**2) / vol / mat.E
        u_z_err_ralative = abs((u_z - u_z_analytic) / u_z)
        assert u_z_err_ralative < 1e-3
        print(f"Simulation approved against analytic.")
    else:
        print(f"Simulation not compared against analytic.")