import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1983)
_bb_ = 0.99999999999 # to shrink point coordinates just very slightly, so that no error raises due to a point not being inside mesh.


def six_cell_mesh_2D(l=10.):
    """
    Returns a fenics mesh object with 6 cells like a Hexagonal.
    """
    a = b = l / 2.
    b = l * (3. ** 0.5) / 2.
    p0 = df.Point(0., 0.)
    p1 = df.Point(l, 0.)
    p2 = df.Point(a, b)
    p3 = df.Point(-a, b)
    p4 = df.Point(-l, 0.)
    p5 = df.Point(-a, -b)
    p6 = df.Point(a, -b)

    mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(mesh, 'triangle', 2, 2) # A 2D triangular mesh
    editor.init_vertices(7)
    editor.init_cells(6)
    editor.add_vertex(0, p0)
    editor.add_vertex(1, p1)
    editor.add_vertex(2, p2)
    editor.add_vertex(3, p3)
    editor.add_vertex(4, p4)
    editor.add_vertex(5, p5)
    editor.add_vertex(6, p6)
    editor.add_cell(0, [0, 1, 2])
    editor.add_cell(1, [0, 2, 3])
    editor.add_cell(2, [0, 3, 4])
    editor.add_cell(3, [0, 4, 5])
    editor.add_cell(4, [0, 5, 6])
    editor.add_cell(5, [0, 6, 1])
    editor.close()
    return mesh


def project_with_custom_dx(f, i_projection, dx_projection):
    """
    Manually projects `f` onto the space `i_projection` using the given `dx_projection` measure.
    """
    v = df.TestFunction(i_projection)
    projected_f = df.Function(i_projection)
    ## Use linear solver:
    u = df.TrialFunction(i_projection)
    a = df.inner(u, v) * dx_projection
    L = df.inner(f, v) * dx_projection
    df.solve(a == L, projected_f)
    ## ALternative:
    ## WAY-1:
    # R = df.inner((projected_f - f), v) * dx_projection
    # K = df.derivative(R, projected_f)
    # df.solve(R == 0, projected_f, J=K)
    return projected_f

def evaluate_basis_functions_highlevel(V, ps):
    """
    Evaluate basis functions of the function space V at arbitrary coordinates ps.
    Returns:
        shape (len(ps), V.dim()) array where each column is the i-th basis function evaluated at all points.
    """
    ndofs = V.dim()
    basis_vals = np.zeros((len(ps), ndofs))
    for i in range(ndofs):
        phi = df.Function(V)
        phi.vector()[i] = 1.0
        for j, x in enumerate(ps):
            basis_vals[j, i] = phi(x)
    return basis_vals

def get_unique_legend_labels_and_handles_of_axes(axes):
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    return dict(zip(labels, handles)) # Keeps the *last* occurrence

def pp_estimations(f, f_est, f_intr, ps_intr, label, _print=False):
    f_est_at_ps_intr = np.array([f_est(_bb_*p) for p in ps_intr])
    _err = np.linalg.norm(f_intr.vector() - f_est_at_ps_intr) / np.linalg.norm(f_intr.vector())
    
    if _print:
        print(f"Estimated nodal values ({label}):")
        print(f_est.vector()[:], '\n')
        print('Interpolation of estimated nodal values (as a discrete field ({label}):')
        print(f_est_at_ps_intr, '\n')

    ps_f = f.function_space().tabulate_dof_coordinates()
    ps_f_est = f_est.function_space().tabulate_dof_coordinates()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.sca(axes[0])
    df.plot(f_cg)
    df.plot(mesh)
    plt.plot(ps_intr[:,0], ps_intr[:,1], marker='.'
             , linestyle='', color='red', label='Data-points')
    for x, val in zip(ps_f, f.vector()):
        axes[0].text(x[0], x[1], f"{val:.2f}", color="black", fontsize=12, ha='center', va='center')
    axes[0].set_title("Ground truth", fontsize=12)
    plt.sca(axes[1])
    df.plot(f_est)
    df.plot(mesh)
    plt.plot(ps_intr[:,0], ps_intr[:,1], marker='.'
             , linestyle='', color='red', label='Data-points')
    for x, val in zip(ps_f_est, f_est.vector()):
        axes[1].text(x[0], x[1], f"{val:.2f}", color="black", fontsize=12, ha='center', va='center')
    axes[1].set_title(f"Estimated ({label})", fontsize=12)
    
    unique = get_unique_legend_labels_and_handles_of_axes(axes)
    fig.legend(unique.values(), unique.keys(), loc='lower center', ncol=3)
    fig.suptitle(f"Uniteless norm-2 error of estimated data = {_err:.2e}.", fontsize=12)
    plt.tight_layout()
    plt.show()
    return f_est_at_ps_intr

if __name__=='__main__':

    ############### USER-LEVEL INPUTs ###############
    data_degree = 4 # over which the discrete field is stored.
    estimation_degree = 1 # for interpolating estimated nodal values, returning estimated values for discrete field
    _nose_level = 0.0 # the noise added to the ground-truth discrete field (percentage-wise)
    _data_points_space = 'cg' # 'cg' or 'q'
    _num_of_mesh_refinement = 1
    _print = False
    #################################################

    ############### COMPUTATIONs ###############

    mesh = six_cell_mesh_2D()
    for i in range(_num_of_mesh_refinement):
        mesh = df.refine(mesh)

    # Lagrange space
    e_cg = df.FiniteElement(family='CG', cell=mesh.ufl_cell()
                            , degree=estimation_degree)
    i_cg = df.FunctionSpace(mesh, e_cg)
    f_cg = df.Function(i_cg)
    f_cg.vector()[:] = abs(np.random.normal(loc=0., scale=0.5, size=i_cg.dim()))
    ps_cg = i_cg.tabulate_dof_coordinates()
    id_center_node = np.where(np.all(ps_cg == np.array([0., 0.]), axis=1))[0][0]
    print(f"ID of the middle node is '{id_center_node}'.\n")
    if _print:
        print('Ground truth nodal values:')
        print(f_cg.vector()[:], '\n')

    if _data_points_space=='cg':
        # CG space over which the discrete field is defined
        e_data = df.FiniteElement(family="CG", cell=mesh.ufl_cell(), \
                                degree=data_degree)
        dxm = df.dx(mesh)
    else:
        # Quadrature space over which the discrete field is defined
        e_data = df.FiniteElement(family="Quadrature", cell=mesh.ufl_cell(), \
                                degree=data_degree, quad_scheme="default")
        # Integral measure
        md = {'quadrature_degree': data_degree, 'quadrature_scheme': 'default'}
        dxm = df.dx(domain=mesh, metadata = md)
        df.parameters["form_compiler"]["quadrature_degree"] = data_degree
    i_data = df.FunctionSpace(mesh, e_data)
    f_data = df.Function(i_data)
    ps_data = i_data.tabulate_dof_coordinates()
    plt.figure()
    plt.plot(ps_data[:,0], ps_data[:,1], linestyle='', marker='.')
    df.plot(mesh)
    plt.title('Mesh and data points')
    plt.show()
    print(f"Number of discrete field points = {i_data.dim()} .\n")
    f_data.vector()[:] = [f_cg(_bb_*p) for p in ps_data]
        # abs(np.random.normal(loc=0., scale=0.5, size=i_data.dim()))
    if _print:
        print('Ground truth discrete field (evaluated/interpolated from ground truth nodal values):')
        print(f_data.vector()[:], '\n')
    if _nose_level!=0.:
        std_noise = np.std(f_data.vector()) * _nose_level
        _noise = np.random.normal(loc=0., scale=std_noise, size=(i_data.dim()))
        f_data.vector()[:] += _noise
        _aa = '\n, with noisy discrete field data'
    else:
        _aa = ''

    ##### ESTIMATION of nodal values (Approach 1, which is NOT correct!) #####
    if estimation_degree!=1:
        _msg = "\n\n\tEstimation with the first approach is only possible for\n\t'estimation_degree=1'"
        _msg += ", because the computation of lumped mass matrix\n\tfor higher order degrees is not implemented here.\n\n"
        print(_msg)
    else:
        f_cg_estimated_1 = df.Function(i_cg)
        v_cg = df.TestFunction(i_cg)
        # _u_cg = df.TrialFunction(i_cg)
        # M_consistent = df.assemble(_u_cg * v_cg * dxm).array()
        # M_lumped = M_consistent.sum(axis=1)
        M_lumped = df.assemble(v_cg * df.Constant(1.0) * dxm).get_local()
        f_cg_estimated_1.vector()[:] = df.assemble(f_data * v_cg * dxm)[:] / M_lumped
        f_cg_est_1_at_ps_data = pp_estimations(f_cg, f_cg_estimated_1
                                               , f_data, ps_data, f"approach-1{_aa}", _print)

    ##### ESTIMATION of nodal values (Approach 2) #####
    f_cg_estimated_2 = df.Function(i_cg)
    N = evaluate_basis_functions_highlevel(i_cg, _bb_ * ps_data)
    A = np.linalg.inv(N.T @ N) @ N.T
    f_cg_estimated_2.vector()[:] = A @ f_data.vector()[:]
    f_cg_est_2_at_ps_data = pp_estimations(f_cg, f_cg_estimated_2
                                           , f_data, ps_data, f"approach-2{_aa}", _print)

    ##### ESTIMATION of nodal values (Approach 3) #####
    f_cg_estimated_3 = df.project(f_data, i_cg)
    f_cg_est_3_at_ps_data = pp_estimations(f_cg, f_cg_estimated_3
                                           , f_data, ps_data, f"approach-3{_aa}", _print)

    f_cg_estimated_4 = project_with_custom_dx(f_data, i_cg, dxm)
    f_cg_est_4_at_ps_data = pp_estimations(f_cg, f_cg_estimated_4
                                           , f_data, ps_data, f"approach-4{_aa}", _print)

    print('\n\tDONE !')