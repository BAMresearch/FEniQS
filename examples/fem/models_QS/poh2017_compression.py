from feniQS.problem.problem import *
from feniQS.problem.time_stepper import *
from feniQS.problem.post_process import *

pth_poh2017_compression = CollectPaths('./examples/fem/models_QS/poh2017_compression.py')
pth_poh2017_compression.add_script(pth_problem)
pth_poh2017_compression.add_script(pth_time_stepper)
pth_poh2017_compression.add_script(pth_post_process)

class FenicsConfig:
    el_family = 'Lagrange'
    shF_degree_u = 1
    shF_degree_ebar = 1
    integ_degree = 2
    
class Geometry2D_Poh_Compression:
    """
    Presents the geometry in fig. (14) of the Poh's [2017] paper
    """
    L = 1. # the length of square
    # for the weakened area on the bottom-left side of square:
    lx_weakened = L/10
    ly_weakened = L/20
    fy_to_AEK0 = - 1.0 # normalized F: force "in y-direction" devided by A * E * K0. Minus sign is due to force being applied in "-y" direction
    uy_to_L = - 0.001 # normalized u in (y) direction as inhomogenous BC at top edge of the rectangle
    

def poh_compression_gK_exp(mesh, geo, e0, ef, alpha, e0_reduction, tol):
    ### First we must assign different values for "e0" inside and outside of weakened area
    
    ### Way 1) direct use of expression
    e0_expr = df.Expression('(x[0] < x_to + tol && x[1] < y_to + tol) ? val_in : val_out', \
                              x_to=geo.lx_weakened, y_to=geo.ly_weakened, val_in=e0_reduction*e0, val_out=e0, tol=tol, degree=0)
        
    ## Way 2) using MeshFunction and domains
    # dom_in = df.CompiledSubDomain('(x[0] <= x_to + tol && x[1] <= y_to + tol)', \
    #                           x_to=geo.lx_weakened, y_to=geo.ly_weakened, tol=tol)
    # dom_out = df.CompiledSubDomain('(x[0] >= x_to - tol || x[1] >= y_to - tol)', \
    #                           x_to=geo.lx_weakened, y_to=geo.ly_weakened, tol=tol)
    # m_f = marked_mesh_function_from_domains([dom_in, dom_out], mesh, dim=mesh.topology().dim(), tp='size_t') # dim=mesh.topology().dim() implies that mesh_function is representing "cells"
    # class E0_Expr(df.UserExpression):
    #     def __init__(self, m_function, e0_vals, **kwargs):
    #         super(E0_Expr, self).__init__(**kwargs)
    #         self.m_function = m_function
    #         self.e0_vals = e0_vals
    #     def eval_cell(self, values, x, cell):
    #         values[0] = self.e0_vals[self.m_function[cell.index]]
    # e0_expr = E0_Expr(m_f, [e0_reduction * e0, e0], degree=0)
    
    return GKLocalDamageExponential(e0=e0_expr, ef=ef, alpha=alpha)

def poh_compression_fenics_2d_force_control(mat, mesh, geo, max_traction, T, fen_config=FenicsConfig, K_current=df.Constant(0.0), f=df.Constant((0.0, 0.0)), tol=1E-14):
    fen = FenicsGradientDamage(mat, mesh, fen_config, K_current)
    
    # The Neumann BCs must be defined before building variationals
    poh_compression_bcs_NM(fen, geo, max_traction, T, tol)
    
    # Discretize and build variational problem
    fen.build_variational_functionals(f=f, integ_degree=fen_config.integ_degree)
    
    poh_compression_bcs_DR(fen, tol)
    
    fen.build_solver(time_varying_loadings=[fen.bcs_NM_tractions[0][1]])
    
    return fen

def poh_compression_fenics_2d_disp_control(mat, mesh, geo, u_max, T, fen_config=FenicsConfig, K_current=df.Constant(0.0), f=df.Constant((0.0, 0.0)), tol=1E-14):
    fen = FenicsGradientDamage(mat, mesh, fen_config, dep_dim=2, K_current=K_current)
    
    # Discretize and build variational problem
    fen.build_variational_functionals(f=f, integ_degree=fen_config.integ_degree)
    
    poh_compression_bcs_DR(fen, tol)
    u_inhomogenous = poh_compression_bcs_DR_inhomogenous(fen, geo, u_max, T, tol)
    
    fen.build_solver(time_varying_loadings=[u_inhomogenous])
    
    return fen

def poh_compression_bcs_DR(fen, tol):
    i_u = fen.i_mix.sub(0)
    def boundary_bot(x, on_boundary):
        return on_boundary and df.near(x[1], 0., tol)
    u_0 = df.Constant(0.0)
    bc_bot, bc_bot_dofs = boundary_condition(i_u.sub(1), u_0, boundary_bot)
    fen.bcs_DR.append(bc_bot)
    fen.bcs_DR_dofs.extend(bc_bot_dofs)
    
    def boundary_00(x, on_boundary):
        return df.near(x[0], 0., tol) and df.near(x[1], 0., tol) # It is crucial to not include on_boundary for pointwise boundary condition
    bc_00, bc_00_dofs = boundary_condition_pointwise(i_u.sub(0), 0., boundary_00) # additional fix in x-direction at (x[0]=0, x[1]=0)
    fen.bcs_DR.append(bc_00)
    fen.bcs_DR_dofs.extend(bc_00_dofs)
    
def poh_compression_bcs_DR_inhomogenous(fen, geo, u_max, T, tol):
    i_u = fen.i_mix.sub(0)
    def boundary_top(x, on_boundary):
        return on_boundary and df.near(x[1], geo.L, tol)
    u_top = df.Expression('u_max * t / T', degree=2, t=0., T=T, u_max=u_max)
    bc_top, bc_top_dofs = boundary_condition(i_u.sub(1), u_top, boundary_top)
    fen.bcs_DR_inhom.append(bc_top)
    fen.bcs_DR_inhom_dofs.extend(bc_top_dofs)
    return u_top
    
def poh_compression_bcs_NM(fen, geo, f_max, T, tol):
    boundary_parts = df.MeshFunction('size_t', fen.mesh, fen.mesh.topology().dim() - 1)
    top_edge  = df.CompiledSubDomain('near(x[1], l, tol)', l=geo.L, tol=tol)
    # class TopEdge(df.SubDomain):
    #     def inside(self, x, on_boundary):
    #         return df.near(x[1], geo.L, tol)
    # top_edge = TopEdge()
    top_edge.mark(boundary_parts, 1)
    ds = df.Measure('ds', domain = fen.mesh, subdomain_data = boundary_parts)
    ds_NM = ds(1) # "1" goes to "top_edge"
    f_top = df.Expression('f_max * t / T', degree=2, t=0., T=T, f_max=f_max)
    f_0 = df.Constant(0.0)
    fen.bcs_NM_tractions.append(df.as_tensor((f_0, f_top)))
    fen.bcs_NM_measures.append(ds_NM)

def poh2017_compression():
    from pathlib import Path
    _name = whoami()
    _path = str(Path(__file__).parent) + '/' + _name + '/'
    make_path(_path)
    logger = LoggerSetup(_path + _name + '.log') # create a logger with log-file of the same name
    
    ############### Parameters / Objects #############################
    tol = 1E-14
    
    geo = Geometry2D_Poh_Compression
    E = 20e3
    nu = 0.2
    k = 1. # of modified Von-mises equivalent strain
    K0 = .0001
    K0_reduction = 0.5 # 50% reduction
    alpha = 0.99
    beta = 100
    l_nonlocal_strain = 0.05 * geo.L
    c_min = l_nonlocal_strain ** 2
    c = 0.
    
    resXY = 80; # same for both x, y directions
    
    T = 1.0 # quasi-static time
    
    eta = 5
    R = 0.005
    
    ################### Problem and Solution #############################
    mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(geo.L, geo.L), resXY, resXY, diagonal='right')
    gK = poh_compression_gK_exp(mesh=mesh, geo=geo, e0=K0, ef=1.0/beta, alpha=alpha, e0_reduction=K0_reduction, tol=tol)
    interaction_function = InteractionFunction_exp(eta=eta, R=R)
    fn_eps_eq = ModifiedMises(nu=nu, k=k)
    mat = GradientDamageConstitutive(E=E, nu=nu, constraint='PLANE_STRAIN', gK=gK, c_min=c_min, c=c \
                                     , interaction_function=interaction_function, fn_eps_eq=fn_eps_eq)
    
    
    ## force-control loading
    # fen = poh_compression_fenics_2d_force_control(mat, mesh, geo, geo.fy_to_AEK0 * E * K0, T, tol=tol)
    ## disp.-control loading
    fen = poh_compression_fenics_2d_disp_control(mat, mesh, geo, geo.uy_to_L * geo.L, T, tol=tol)
    
    # write the mesh to a file (suitable for being imported e.g. to Paraview)
    write_to_xdmf(fen.mesh, xdmf_name=_name+'_mesh.xdmf', xdmf_path=_path)
    
    pp = PostProcessGradientDamage(fen, _name, _path, reaction_dofs=[fen.bcs_DR_inhom_dofs])
    ts = TimeStepper(fen.solve, pp, logger=logger.file, solution_field=fen.u_mix, \
                     increase_num_iter=8, increase_factor=1.2)
    t_end = T
    dt = 0.01
    # iterations = ts.equidistant(t_end, dt)
    iterations = ts.adaptive(t_end=t_end, dt=dt)
    
    fig = plt.figure()
    plt.plot(ts.ts, iterations)
    plt.title('Number of iterations over time\nSum = ' + str(sum(iterations)) + ', # time steps = ' + str(len(ts.ts)))
    plt.xlabel('t')
    plt.ylabel('Iterations number')
    plt.savefig(_path + 'iterations.pdf')
    
    pp.plot_reaction_forces(['Normalized reaction force'], full_file_names=[_path + 'reaction_force.pdf'], factor=-1/(E*K0))
    pp.plot_residual_at_free_DOFs('Normalized residual-norm of free DOFs', full_file_name=_path + 'residual_norm_free_DOFs.pdf', factor=1/(E*K0))
    
    df.list_timings(df.TimingClear.keep, [df.TimingType.wall])
    
    print('One FEniCS problem of "' + _name + '" was solved.')
        
    
if __name__ == "__main__":
    df.set_log_level(20)
    poh2017_compression()