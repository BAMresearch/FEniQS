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
    integ_degree = 2 # Regarding Quadrature space that stores stresses
    ss_dim = 3 # For: sigma_xx, sigma_yy, sigma_xy
    n_refinemensts = 3

    ### MESH ###
    _scale = 1.0 # for geometry
    mesh = one_cell_mesh_2D(0., 0., _scale*20., _scale*10., -_scale*5., _scale*20.) # A mesh with ONE single cell
    for i in range(n_refinemensts):
        mesh = df.refine(mesh)
    metadata = {"quadrature_degree": integ_degree, "quadrature_scheme": "default"}
    dxm = df.dx(domain=mesh, metadata=metadata)
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

    ### QUADRATIC ERROR "Q = Integral ( Sigma : S : Sigma )" ###
        ## Space (scalar) for damage parameter at GPs
    elem_GP = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=integ_degree \
                                            , quad_scheme="default")
    i_GP = df.FunctionSpace(mesh, elem_GP)
    nGPs = i_GP.dim() # get total number of gauss points
        ## Space (tensor) for "inverse tanget operator" (S) at GPs
    elem_tensor = df.TensorElement("Quadrature", mesh.ufl_cell() \
                                   , degree=integ_degree, shape=(ss_dim, ss_dim) \
                                    , quad_scheme="default")
    i_tensor = df.FunctionSpace(mesh, elem_tensor)
    S_GP = df.Function(i_tensor, name="S: Inverse tangent operator at GPs")

        ## Assign arbitrary inverse tangent operator (with random damage) at GPs
    damage_level = 0.1
    damages = abs(np.random.normal(0., damage_level, (nGPs,))) # strictly positive
    damages = np.array([min(0.99, d) for d in damages]) # Upper bound = 0.99
    S_GP_num = np.array([[4., -2., 1.], [-2., 8., -1.], [1., -1., 3.]])**(-1)
    S_GPs_num = np.concatenate([d * S_GP_num.flatten() for d in damages])
    S_GP.vector().set_local(S_GPs_num.flatten())

        ## Compute Q by direct integration
    Q1 = df.assemble(df.inner(q_ss, df.dot(S_GP, q_ss)) * dxm)

        ## Compute Q via an assembled S_matrix
    sig_trial = df.TrialFunction(i_ss)
    sig_test = df.TestFunction(i_ss)
    S_form = df.inner(sig_trial, df.dot(S_GP, sig_test)) * dxm
    if integ_degree>1:
        df.parameters["form_compiler"]["representation"] = "quadrature"
    S_matrix = df.assemble(S_form).array()
    df.parameters["form_compiler"]["representation"] = "uflacs"
    Q2 = np.einsum('m,mn,n->', q_ss.vector().get_local() \
                   , S_matrix, q_ss.vector().get_local())
    
        ## Compare Q1 and Q2
    err_r_Q = abs((Q1 - Q2) / Q1)
    _msg = f"\nQuadratic Q = integral(sigma:S:sigma), computed by:"
    _msg += f"\n\tdirect integration    = {Q1:.5f} ,"
    _msg += f"\n\tmatrix multiplication = {Q2:.5f} ."
    _msg += f"\nRelative error = {err_r_Q:.2e} ."
    print(_msg)
    assert (err_r_Q < 1e-12)

    print("\n----- DONE! -----") 