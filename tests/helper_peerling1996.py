from feniQS.general.general import *
from feniQS.fenics_helpers.fenics_functions import *

pth_helper_peerling1996 = CollectPaths('helper_peerling1996.py')
pth_helper_peerling1996.add_script(pth_general)
pth_helper_peerling1996.add_script(pth_fenics_functions)

class AnalyticSolutionPeerlingPerfectDamage: # of the Peerlings paper (1996)
    def __init__(self, gK, geometry, c, _print=False):
        self.gK = gK
        self.geo = geometry
        self.c = c
        self.calculate_coeffs(_print)

    def calculate_coeffs(self, _print):
        """
        The analytic solution is following Peerlings paper (1996) but with
        b(paper) = b^2   (here)
        g(paper) = g^2   (here)
        c(paper) = l^2   (here)
        l(paper) = geo.W (here)
        This modification eliminates all the sqrts in the formulations.
        Plus: the formulation of the GDM in terms of l ( = sqrt(c) ) is 
        more common in modern pulbications.
        """

        # imports only used here...
        from sympy import Symbol, symbols, N, integrate, cos, exp, lambdify
        import scipy.optimize
        import math

        # unknowns
        x = Symbol("x")
        unknowns = symbols("A1, A2, B1, B2, C, b, g, w")
        A1, A2, B1, B2, C, b, g, w = unknowns

        l = math.sqrt(self.c)
        kappa0 = self.gK.K0  # consistent with a PERFECT-damage model

        # 0 <= x <= W/2
        e1 = C * cos(g / l * x)
        # W/2 <  x <= w/2
        e2 = B1 * exp(b / l * x) + B2 * exp(-b / l * x)
        # w/2 <  x <= L/2
        e3 = A1 * exp(x / l) + A2 * exp(-x / l) + (1 - b * b) * kappa0

        de1 = e1.diff(x)
        de2 = e2.diff(x)
        de3 = e3.diff(x)

        W = self.geo.W
        L = self.geo.L
        deltaL = self.geo.deltaL
        alpha = self.geo.alpha

        eq1 = N(e1.subs(x, W / 2) - e2.subs(x, W / 2))
        eq2 = N(de1.subs(x, W / 2) - de2.subs(x, W / 2))
        eq3 = N(e2.subs(x, w / 2) - kappa0)
        eq4 = N(de2.subs(x, w / 2) - de3.subs(x, w / 2))
        eq5 = N(e3.subs(x, w / 2) - kappa0)
        eq6 = N(de3.subs(x, L / 2))
        eq7 = N((1 - alpha) * (1 + g * g) - (1 - b * b))
        eq8 = N(
            integrate(e1, (x, 0, W / 2))
            + integrate(e2, (x, W / 2, w / 2))
            + integrate(e3, (x, w / 2, L / 2))
            - deltaL / 2
        )

        equations = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]

        python_functions = []
        for eq in equations:
            python_functions.append(lambdify(unknowns, eq))

        def global_func(x):
            result = np.zeros(8)
            for i in range(8):
                result[i] = python_functions[i](*x)

            return result

        result = scipy.optimize.root(
            global_func, [0.0, 5e2, 3e-7, 7e-3, 3e-3, 3e-1, 2e-1, 4e1]
        )
        if _print:
            print(result)
        if not result["success"]:
            raise RuntimeError(
                "Could not find the correct coefficients. Try to tweak the initial values."
            )
        self.coeffs = result["x"]

    def eval(self, x):
        import math

        A1 = self.coeffs[0]  # 2.09844964e-41
        A2 = self.coeffs[1]  # 5.64087844e+02
        B1 = self.coeffs[2]  # 3.18822675e-07
        B2 = self.coeffs[3]  # 7.30855455e-03
        C = self.coeffs[4]  # 3.38459683e-03
        b = self.coeffs[5]  # 2.60666136e-01
        g = self.coeffs[6]  # 1.88718384e-01
        w = self.coeffs[7]  # 3.64691602e+01

        l = math.sqrt(self.c)
        kappa0 = self.gK.K0  # consistent with a PERFECT-damage model

        if x <= self.geo.W / 2.0:
            value = C * np.cos(g / l * x)
        elif x <= w / 2.0:
            value = B1 * np.exp(b / l * x) + B2 * np.exp(-b / l * x)
        else:
            value = (1.0 - b * b) * kappa0 + A1 * np.exp(x / l) + A2 * np.exp(-x / l)
        return value
    
class DispEbarKappaDamageOverX_ReactionForces_GradientDamage:
    def __init__(self, fen, X, u_xdmf_name, u_xdmf_path=None, reaction_dofs=None):
        self.fen = fen
        self.X = X # Every entry is the full coordinates (1D or 2D or 3D) of the postprocessed points
        self.reaction_dofs = reaction_dofs # for which, the reaction force (residual) will be calculated
        
        # Every entry at any of the following lists is for a certain time ("t") and is a list of values over the given X / reaction_dofs
        self.u = [] # Displacement;
        self.Ebar = [] # Nonlocal equivalent strain
        self.K = [] # Internal variable (Kappa)
        self.D = [] # Damage parameter
        self.reaction_forces = []
        
        self.ts = [] # list of times
        
        if u_xdmf_path is None:
            u_xdmf_path = './'
        make_path(u_xdmf_path)
        self.u_xdmf = df.XDMFFile(u_xdmf_path + u_xdmf_name)

    def __call__(self, t, logger, write_residual_vector=False):
        self.ts.append(t)
        u_plot, ebar_plot = self.fen.u_mix.split()
        u_vals = [u_plot(Xi) for Xi in self.X]
        self.u.append( u_vals )
        Ebar_vals = [ebar_plot(Xi) for Xi in self.X]
        self.Ebar.append( Ebar_vals )
        K_vals = [self.fen.u_K_current(Xi) for Xi in self.X]
        self.K.append( K_vals )
        D_vals = [self.fen.mat.gK.g_eval(K) for K in K_vals]
        self.D.append( D_vals )
        
        reaction_force, _ = compute_residual(F=self.fen.F_u+self.fen.F_ebar \
                                             , bcs_dofs=self.fen.bcs_DR_dofs + self.fen.bcs_DR_inhom_dofs \
                                             , reaction_dofs=[self.reaction_dofs], logger=logger, write_residual_vector=write_residual_vector)
        self.reaction_forces.append(reaction_force[0])
        
        u_plot.rename("u", "u")
        self.u_xdmf.write(u_plot, t)
        
        if logger is not None:
            logger.debug('\n******************** Results at t = ' + "%.2f"%t + ' :' )
            logger.debug('_____Displacement_____\n' + str(u_vals))
            logger.debug('_____E_bar____________\n' + str(Ebar_vals))
            logger.debug('_____Damage___________\n' + str(D_vals))
        
    def plot_over_x0(self, tit, legs, ts=-1, sol_analytic=None, x_plot=None, u_id=0, full_file_name=None):
        """
        sol_analytic must have the members that are used in this method: X, D and Ebar
        """
        if x_plot is None:
            x_plot = self.X
        fig1, _ = plt.subplots(nrows=2, ncols=2)
        fig1.suptitle(tit, fontsize=14)
        fig1.tight_layout(pad=5.0)
        fig1.subplots_adjust(top=0.8)
        
        if type(self.u[ts][0]) is np.ndarray:
            u_plot = [u[u_id] for u in self.u[ts]] # only in x_direction
        else:
            u_plot = self.u[ts]
        ax1 = plt.subplot(221)
        plt.plot(x_plot, u_plot, label=legs[0])
        plt.title('Displacement')
        plt.xlabel('x')
        plt.ylabel('u')
        ax1.legend(prop={"size":7})
        
        ax2 = plt.subplot(222)
        plt.plot(x_plot, self.Ebar[ts], label=legs[0])
        if sol_analytic is not None:
            plt.plot(sol_analytic.X, sol_analytic.Ebar, label=legs[1])
        plt.title('Nonlocal equivalent strain')
        plt.xlabel('x')
        plt.ylabel('Epsilon_bar')
        ax2.legend(prop={"size":7})
        
        ax3 = plt.subplot(223)
        plt.plot(x_plot, self.K[ts], label=legs[0])
        plt.title('Internal variable')
        plt.xlabel('x')
        plt.ylabel('Kappa')
        ax3.legend(prop={"size":7})
        
        ax4 = plt.subplot(224)
        plt.plot(x_plot, self.D[ts], label=legs[0])
        if sol_analytic is not None:
            plt.plot(sol_analytic.X, sol_analytic.D, label=legs[1])
        plt.title('Damage parameter')
        plt.xlabel('x')
        plt.ylabel('D')
        ax4.legend(prop={"size":7})
        
        # plt.show()
        if full_file_name is not None:
            plt.savefig(full_file_name)
    
    def plot_reaction_forces(self, tit, dof='sum', full_file_name=None):
        fig1 = plt.figure()
        if dof=='sum':
            f_dof = [sum(f) for f in self.reaction_forces]
        else:
            f_dof = [f[dof] for f in self.reaction_forces]
        plt.plot(self.ts, f_dof)
        plt.title(tit)
        plt.xlabel('t')
        plt.ylabel('f')
        
        # plt.show()
        if full_file_name is not None:
            plt.savefig(full_file_name)
            
    def close_files(self):
        pass