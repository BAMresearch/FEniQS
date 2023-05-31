from feniQS.structure.struct_rectangle2D import *
from feniQS.structure.struct_peerling1996 import *
from feniQS.problem.QSM_GDM import *
import unittest

class TestPenaltyApproach(unittest.TestCase):
    def test_cantilever2d(self):
        pars = ParsRectangle2D()
        pars.res_x = 1; pars.res_y = 1
        pars._plot = False
        struct = Rectangle2D(pars)
        pars_gdm = GDMPars()
        pars_gdm.e0_min = 1.0 # Pure elastic
        pars_gdm.c_min = (max((pars.lx / pars.res_x), (pars.ly / pars.res_y)) * 1.5) ** 2
        pars_gdm.constraint = 'PLANE_STRAIN'
        model = QSModelGDM(pars_gdm, struct)
        solve_options = QuasiStaticSolveOptions(solver_options=get_fenicsSolverOptions())
        solve_options.reaction_places = ['bot_left_y', 'bot_right_y', 'top_left_y', 'top_right_y']
        solve_options._plot = False
        model.solve(solve_options)
        points = model.struct.mesh.coordinates()
        pp0 = model.pps[0]
        us = pp0.eval_checked_u(points)[-1] # Crucial to get results of model before modifying it, since the same post-processor object is used and overwritten.
        
        ## Penalty approach
        dy = struct.pars.loading_level
        pp0 = model.pps[0]
        f_top = sum(pp0.eval_checked_reaction_forces()[-1][-1]) # Indices go to: reaction place, load-steps (here, only 1)
        K_top = f_top / dy / 2. # Devided by 2, because we have 2 * penalty_weight in FenicsGradientDamage
        top_nodes = np.array(struct.get_reaction_nodes(['top_left', 'top_right']))
        top_nodes = [r.flatten() for r in top_nodes]
        top_us = pp0.eval_checked_u(top_nodes)[-1]
        top_us_data = top_us[:, 1] + 0. # deepcopy
        top_us_data[:] += dy
        top_us_data = [top_us_data] # for one checkpoint
        top_dofs_y = model.get_reaction_dofs(['top_left_y', 'top_right_y'])
        top_dofs_y = [d[0] for d in top_dofs_y]
        
        model_by_data = model # We only set new 'penalty_dofs' to make the model data_driven.
        model_by_data._name += '_dataDriven'
        model_by_data.penalty_dofs = top_dofs_y
        model_by_data.penalty_weight.assign(K_top)
        
        model_by_data.fen.bcs_DR_inhom = []
        model_by_data.fen.bcs_DR_inhom_dofs = []
        if solve_options.t_end not in solve_options.checkpoints:
            solve_options.checkpoints = solve_options.checkpoints + [solve_options.t_end]
        model_by_data.build_solver(solve_options) ## IMPORTANT: Since we modified the inhomogeneous boundary-condistions and the model is not fresh (i.e. built_solver=True), we must re-build solver.
        funcs = [] # The initial guess for solution is regardless of data (comes from past converged solution)
        funcs.append(model_by_data.fen.u_data)
        data_feeder = DolfinFunctionsFeeder(funcs, dofs=top_dofs_y)
        data_prep = ValuesEvolverInTime(data_feeder, values=top_us_data, checkpoints=solve_options.checkpoints)
        model_by_data.prep_methods=[data_prep]
        model_by_data.solve(solve_options)
        
        ## Comparison and Check
        b = True
        pp0_by_data = model_by_data.pps[0]
        us_by_data = pp0_by_data.eval_checked_u(points)[-1]
        _err = np.linalg.norm(us - us_by_data)
        b = b and (np.linalg.norm(_err) < 1e-12)
        
        self.assertTrue(b)

    def test_static_balance_bar1D(self):
        pars = ParsPeerling()
        pars._res = 2
        pars._plot = False
        struct = Peerling1996(pars)
        pars_gdm = GDMPars()
        pars_gdm.constraint = 'UNIAXIAL'
        pars_gdm.e0_min = 1.0 # Pure elastic
        pars_gdm.c_min = (1.5 * pars.L / pars._res) ** 2
        model = QSModelGDM(pars_gdm, struct)
        solve_options = QuasiStaticSolveOptions(solver_options=get_fenicsSolverOptions())
        solve_options.reaction_places = ['left', 'right']
        solve_options._plot = False
        model.solve(solve_options)
        pp0 = model.pps[0]
        fs = pp0.eval_checked_reaction_forces()
        
        p_middle = np.array([pars.L/2])
        u_middle = pp0.eval_checked_u([p_middle])
        u_middle = [2*uu for uu in u_middle]
        u_middle = np.atleast_3d(u_middle)
        dofs_middle = dofs_at([p_middle], model.fen.get_i_full(), model.fen.get_iu())
        
        model_by_data = model # We only set new 'penalty_dofs' to make the model data_driven.
        model_by_data._name += '_dataDriven'
        model_by_data.penalty_dofs = dofs_middle
        model_by_data.penalty_weight.assign(1e2)
        
        if solve_options.t_end not in solve_options.checkpoints:
            solve_options.checkpoints = solve_options.checkpoints + [solve_options.t_end]
        funcs = [] # The initial guess for solution is regardless of data (comes from past converged solution)
        funcs.append(model_by_data.fen.u_data)
        data_feeder = DolfinFunctionsFeeder(funcs, dofs=dofs_middle)
        data_prep = ValuesEvolverInTime(data_feeder, values=u_middle, checkpoints=solve_options.checkpoints)
        model_by_data.prep_methods=[data_prep]
        model_by_data.solve(solve_options)
        pp0_by_data = model.pps[0]
        fs_by_data = np.array(pp0_by_data.eval_checked_reaction_forces())
        
        ## Balance of the summation of reaction forces with penalty forces
        penalty_forces = model_by_data.fen.penalty_forces[0] # It is like external forces
        f_total = np.sum(fs_by_data) # Sum of reaction forces at supports
        zz = f_total - penalty_forces
        assert abs(zz) < 1e-12

if __name__ == "__main__":
    df.set_log_level(30)
    unittest.main()
    df.set_log_level(20)


