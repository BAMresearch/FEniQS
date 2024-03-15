from feniQS.structure.helper_mesh_fenics import *
import matplotlib.pyplot as plt

def visualize_projected_stress(q_ss, mesh, DG_degree):
    ### PROJECT STRESS on DG-space ###
    elem_ss_DG = df.VectorElement(family="DG", cell=mesh.ufl_cell(), \
                         degree=DG_degree, dim=ss_dim, quad_scheme="default")
    i_ss_DG = df.FunctionSpace(mesh, elem_ss_DG)
    form_compiler_parameters = {"quadrature_degree":integ_degree, "quad_scheme": "default"}
    f_ss_DG = df.project(v=q_ss, V=i_ss_DG, form_compiler_parameters=form_compiler_parameters)

    ### PLOT ###
    for ss_comp_id, ss_comp in zip([0,1,2], ['xx','yy','xy']):
        fig = plt.figure()
        c = df.plot(f_ss_DG.sub(ss_comp_id))
        # c = df.plot(f_ss_DG.split()[ss_comp_id])
        plt.title(f"Strees component {ss_comp} (projected)")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(c, cax=cbar_ax)
        plt.show()

if __name__=='__main__':
    shF_degree = 4 # Concerning virtual displacement
    integ_degree = 3 # Regarding Quadrature space that stores stresses
    ss_dim = 3 # For: sigma_xx, sigma_yy, sigma_xy

    ### MESH ###
    _scale = 1.0 # for geometry
    mesh = one_cell_mesh_2D(0., 0., _scale*20., _scale*10., -_scale*5., _scale*20.) # A mesh with ONE single cell
    plt.figure()
    df.plot(mesh)
    plt.title("A single element")
    plt.show()

    ### ARBITRARY STRESSes at GAUSS-POINTs ###
    assert ss_dim==3
    elem_ss = df.VectorElement(family='Quadrature', cell=mesh.ufl_cell()\
                            , degree=integ_degree, dim=ss_dim, quad_scheme="default")
    i_ss = df.FunctionSpace(mesh, elem_ss)
    q_ss = df.Function(i_ss)
    if integ_degree==1:
        q_ss.vector()[:] = [125, -52, 632]
    elif integ_degree==2:
        q_ss.vector()[:] = [125, -52, 632, 1036, -899, 21, 95, -995, 11]
    elif integ_degree==3:
        q_ss.vector()[:] = [125, -52, 632, 1036, -899, 21, 95, -995, 11] \
                        + [-1025, 0, -540, 13, -228, 63, 902, -12995, -1]
    else:
        raise NotImplementedError(f"Example supports 'integ_degree' up to 3.")

    ### VISUALIZATION (of stresses) ??? ###
    visualize_projected_stress(q_ss, mesh, DG_degree=integ_degree)

    ### NODAL INTERNAL FORCEs (entire vector) ###
    elem_u = df.VectorElement(family='CG', cell=mesh.ufl_cell() \
                            , degree=shF_degree, dim=2)
    i_u = df.FunctionSpace(mesh, elem_u)
    u_ = df.TestFunction(i_u)
    def eps(v):
        e = df.sym(df.grad(v))
        return df.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])
    metadata = {"quadrature_degree": integ_degree, "quadrature_scheme": "default"}
    dxm = df.dx(domain=mesh, metadata=metadata)
    f_int = df.inner(eps(u_), q_ss) * dxm # internal forces
    f_int_local = df.assemble_local(f_int, df.Cell(mesh, 0))

    ### EXTRACT f_int_x, f_int_y ###
    dofs_cell = list(i_u.dofmap().cell_dofs(0))
    dofs_x = i_u.sub(0).dofmap().dofs()
    dofs_y = i_u.sub(1).dofmap().dofs()
    ids_x = [dofs_cell.index(i) for i in dofs_x]
    ids_y = [dofs_cell.index(i) for i in dofs_y]
    f_int_x = f_int_local[ids_x]
    f_int_y = f_int_local[ids_y]

    ### COMPUTE ROTATIONAL MOMENTUM ###
    cs = i_u.tabulate_dof_coordinates()
    cs_x = cs[dofs_x, :]
    cs_y = cs[dofs_y, :]
    rot = 0. # rotational momentum
    for f, r in zip(f_int_x, cs_x):
        rot += - f * r[1] # clockwise
    for f, r in zip(f_int_y, cs_y):
        rot += f * r[0] # counter-clockwise

    ### PRINTs ###
    max_force = np.max(abs(f_int_local))
    print(f"Local internal forces: {f_int_local}.\nMaximum absolute value = {max_force:.2e}.\n")
    print(f"Sum of forces (fx, fy) = ({sum(f_int_x):.1e}, {sum(f_int_y):.1e})")
    print(f"Rotational momentum = {rot:.1e}")

    ### CHECK STATIC BALANCE ###
    tol_f = max_force * 1e-14 # relative to the maximum force
        # Translational
    assert abs(sum(f_int_x)) < tol_f
    assert abs(sum(f_int_y)) < tol_f
    tol_r = max_force * mesh.hmax() * 1e-14
        # Rotational
    assert abs(rot) < tol_r

    print("\n----- DONE! -----")