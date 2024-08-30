from feniQS.structure.helper_mesh_fenics import *
import matplotlib.pyplot as plt

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

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
    np.random.seed(1983)
    shF_degree = 2 # Concerning virtual displacement
    integ_degree = 2 # Regarding Quadrature space that stores stresses
    ss_dim = 3 # For: sigma_xx, sigma_yy, sigma_xy
    n_refinemensts = 2

    ### MESH ###
    _scale = 1.0 # for geometry
    mesh = one_cell_mesh_2D(0., 0., _scale*20., _scale*10., -_scale*5., _scale*20.) # A mesh with ONE single cell
    for i in range(n_refinemensts):
        mesh = df.refine(mesh)
    plt.figure()
    df.plot(mesh)
    plt.title("2D lements")
    plt.show()

    ### ARBITRARY STRESSes at GAUSS-POINTs ###
    assert ss_dim==3
    elem_ss = df.VectorElement(family='Quadrature', cell=mesh.ufl_cell()\
                            , degree=integ_degree, dim=ss_dim, quad_scheme="default")
    i_ss = df.FunctionSpace(mesh, elem_ss)
    q_ss = df.Function(i_ss)
    q_ss.vector()[:] = 100. * (2 * np.random.rand(i_ss.dim()) - 1.) # some random stress field

    ### VISUALIZATION (of stresses) ??? ###
    if integ_degree > 1:
        visualize_projected_stress(q_ss, mesh, DG_degree=integ_degree)

    ### NODAL INTERNAL FORCEs (entire vector) from stress vector ###
        ## (1) Via df.assemble directly ##
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
    f_int_assembled = df.assemble(f_int).get_local()
        
        ## (2) Via an assembled matrix to be pre-multiplied by stress vector ##
    assembler_matrix = df.inner(eps(u_), df.TrialFunction(i_ss)) * dxm
    if integ_degree>1:
        df.parameters["form_compiler"]["representation"] = "quadrature"
    assembler_matrix_assembled = df.assemble(assembler_matrix).array()
    df.parameters["form_compiler"]["representation"] = "uflacs"
    f_int_assembled_2 = assembler_matrix_assembled @ q_ss.vector().get_local()
    print(f"Assembly matrix for internal forces:\n\t{assembler_matrix_assembled}\n.")
    plt.figure()
    plt.imshow(assembler_matrix_assembled)
    plt.title("Assembly matrix for internal forces\n(pre-multiplied with stress vector)")
    plt.xlabel('DOF')
    plt.ylabel('DOF')
    plt.colorbar()
    plt.show()

    ### PLOT & CHECK ###
    plt.figure()
    plt.plot(f_int_assembled, linestyle='', marker='.' \
             , color='blue', label='by direct assembly')
    plt.plot(f_int_assembled_2, linestyle='', marker='o' \
             , fillstyle='none', color='red', label='by pre-multiplied matrix')
    plt.legend()
    plt.title("Nodal internal forces from an arbitrary stress field")
    plt.xlabel("DOF")
    plt.ylabel("Internal force")
    plt.show()
    err_r_f = np.linalg.norm(f_int_assembled - f_int_assembled_2) / np.linalg.norm(f_int_assembled)
    print(f"Relative norm-2-error in computed internal forces:\n\t{err_r_f:.2e}")
    assert (err_r_f < 1e-12)

    print("\n----- DONE! -----") 