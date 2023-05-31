from feniQS.problem.problem import *
from feniQS.problem.time_stepper import *
from tests.helper_peerling1996 import *
import unittest

class FenicsConfig:
    el_family = 'Lagrange'
    shF_degree_u = 2
    shF_degree_ebar = 2
    integ_degree = 3
    
class Geometry2D_Peerling:
    L = 100.0
    Ly = 2.0
    W = 10.0  # length of reduced area
    alpha = 0.1  # cross-section reduction factor
    deltaL = 0.05  # Applied disp. load, half of which at each end of bar
        
def peerling_fenics_2d(mat, mesh, geo=Geometry2D_Peerling, fen_config=FenicsConfig, K_current=df.Constant(0.0), f=df.Constant((0.0, 0.0))):
    fen = FenicsGradientDamage(mat, mesh, fen_config, dep_dim=2, K_current=K_current)
    
    # Define "sc_E" for scaling material E modulus (or equivalently, sigma)
    x_from = 0.5 * (geo.L - geo.W)
    x_to = 0.5 * (geo.L + geo.W)
    sc_E = df.Expression('(x[0] >= x_from && x[0] <= x_to) ? sc : 1.0', x_from=x_from, x_to=x_to, sc=(1.0 - geo.alpha), degree=0)
    # x[1]:=y does not matter here
    
    # Discretize and build variational problem
    fen.build_variational_functionals(f=f, integ_degree = fen_config.integ_degree, expr_sigma_scale=sc_E, hist_storage='ebar')
    
    return fen

class TestFenicsGradientDamage_Peerling1996_2D(unittest.TestCase):
    
    def test_peerling_2d_noo0(self):
        dfll = df.get_log_level() # as backup
        df.set_log_level(30)
        
        test_name = whoami()
        test_path = f"./tests/{test_name}/"
        make_path(test_path)
        logger = LoggerSetup(test_path + test_name + '.log') # create a logger with log-file of the same name as test_method
        
        ############### Parameters / Objects #############################
        tol = 1E-14
        
        geo = Geometry2D_Peerling
        L = geo.L
        Ly = geo.Ly
        res_total = 1
        resX = res_total * int(L) # Mesh resolution at x-dir
        resY = res_total * int(Ly) # Mesh resolution at y-dir
        mesh = df.RectangleMesh(df.Point(0, 0), df.Point(L, Ly), resX, resY, diagonal='right')

        max_u_bc = geo.deltaL
        E = 20000.  # E modulus
        K0 = 1e-4  # Damage law (Perfect damage) parameter
            # gradient parameter
        c_min = (1.2 * mesh.hmax()) ** 2
        c = 0.0
        
        T = 1.0 # quasi-static time
        
        gK = GKLocalDamagePerfect(K0)
        
        res_pp = 50 # The resolution for post-processing
        
        plot = True
        
        ######################################################################
        
        # ################### Analytical Solution ##############################
        if type(gK)==GKLocalDamagePerfect:
            sol_analytic = AnalyticSolutionPeerlingPerfectDamage(gK=gK, geometry=geo, c=c_min+c)
            sol_analytic.X = np.linspace(0.0, L, res_pp) # for plot
            xi_to_eval = np.linspace(- L / 2, L / 2, res_pp) # for evaluation (concerning symmetry in the implementation of the analyticSolution class)
            sol_analytic.Ebar = [
                sol_analytic.eval(abs(xi)) for xi in xi_to_eval
            ] # use "abs" due to symmetry
            sol_analytic.D = [gK.g_eval(k) for k in sol_analytic.Ebar]
        
        ################### Problem and Solution #############################
        mat = GradientDamageConstitutive(E=E, nu=0., constraint='PLANE_STRAIN', gK=gK, c_min=c_min, c=c)
        geo.resX = resX
        geo.resY = resY
        fen = peerling_fenics_2d(mat, mesh, geo)
        # write the mesh to a file (suitable for being imported e.g. to Paraview)
        write_to_xdmf(fen.mesh, xdmf_name='mesh.xdmf', xdmf_path=test_path)
        
        ### Define boundary conditions of i_u=i_mix.sub(0)
        i_u = fen.i_mix.sub(0)
        def boundary_left(x):
            return df.near(x[0], 0., tol)
        def boundary_right(x):
            return df.near(x[0], L, tol)
        ## fix at both x,y at x[0]=0
        u_left = df.Constant((0.0, 0.0))
        bc_left, bc_left_dofs = boundary_condition(i_u, u_left, boundary_left)
        u_right = df.Expression('max_u_bc * t / T', degree=2, t=0., T=T, max_u_bc=max_u_bc)
        bc_right, bc_right_dofs = boundary_condition(i_u.sub(0), u_right, boundary_right)
        fen.bcs_DR.append(bc_left)
        fen.bcs_DR.append(bc_right)
        fen.bcs_DR_dofs.extend(bc_left_dofs)
        fen.bcs_DR_dofs.extend(bc_right_dofs)
        
        reaction_bc = fen.bcs_DR[1] # at right-hand side of bar
        reaction_dofs = [key for key in reaction_bc.get_boundary_values().keys()]
        
        fen.build_solver(time_varying_loadings=[u_right])
        
        y_post_process = geo.Ly / 2 # A fix value of y=x[1], for which all plots and results in log-file will be calculated
        x0_pp = np.linspace(0.0, L, res_pp) # in x direction
        x_pp = [[x, y_post_process] for x in x0_pp]
        pp = DispEbarKappaDamageOverX_ReactionForces_GradientDamage(fen, x_pp, 'u.xdmf', test_path, reaction_dofs)
        ts = TimeStepper(fen.solve, [pp], logger=logger.file)
        t_end = T
        dt = 0.04
        ts.equidistant(t_end, dt)
        
        if plot:
            tit = 'Results at y = ' + str(y_post_process) + ' with ' + str(fen.n_elements) + ' elements\nt = ' + str(t_end)
            legs = ['FEniCS-2D', 'Analytic-1D']
            file_name = test_name + '_results.pdf'
            full_file_name = test_path + file_name
            if type(gK)==GKLocalDamagePerfect:
                pp.plot_over_x0(tit, legs, ts=-1, sol_analytic=sol_analytic, x_plot=x0_pp, full_file_name=full_file_name)
            else:
                pp.plot_over_x0(tit, legs, ts=-1, x_plot=x0_pp, full_file_name=full_file_name)
            tit2 = 'Reaction force over time (with ' + str(fen.n_elements) + ' elements)'
            file_name2 = test_name + '_reactionForce.pdf'
            full_file_name2 = test_path + file_name2
            pp.plot_reaction_forces(tit=tit2, full_file_name=full_file_name2)
            
        ### Assert the agreement of analytical solution and FEniCS solution
        if type(gK)==GKLocalDamagePerfect:
            tol_assert = 0.005
            for (analytic, fem) in zip(sol_analytic.D, pp.D[-1]):
                self.assertAlmostEqual(analytic, fem, delta=tol_assert)
            for (analytic, fem) in zip(sol_analytic.Ebar, pp.Ebar[-1]):
                self.assertAlmostEqual(analytic, fem, delta=tol_assert)
        
        df.set_log_level(dfll)
    
if __name__ == "__main__":    
    unittest.main()
