import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import ufl
from feniQS.problem.time_stepper import *
from feniQS.fenics_helpers.fenics_functions import compute_residual

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid") # style must be one of: white, dark, whitegrid, darkgrid, ticks
except ModuleNotFoundError:
    print(f"\n\n\t{'-' * 70}\n\tWARNING: It is recommended to install 'seaborn' to get nicer plots.\n\t{'-' * 70}\n\n")

## All Pars:
E = 20e3
L = 100
W = 10  # length of the bar with reduced E
_res = 200 ## Mesh resolution
loading = 0.03
T = 1.0
e0 = df.Constant(1e-4)
ef = 100e-4
c = 1.0
# damage_law = 'perfect'
damage_law = 'linear'
# damage_law = 'exp'
x_from = (L - W) / 2
x_to = (L + W) / 2
# Imperfection by having smaller cross section (or Young modulus):
red_factor = 0.9
K0_middle = 0.0
# Imperfection by initiating pre-damage:
# red_factor = 1.0
# K0_middle = 5.0 * self.e0
solver_type = "newton"
solver_tol = 1e-11
el_family = 'Lagrange'
shF_degree_u = 2
shF_degree_ebar = 2
integ_degree = 3

class GKLocalDamageExponential:
    def __init__(self, e0, ef, alpha=1.):
        self.alpha = alpha
        self.e0 = e0
        self.ef = ef
    def g(self, K):
        condition = df.lt(K, self.e0)
        return df.conditional(condition, 0, 1 - self.e0 * (1 - self.alpha + self.alpha * df.exp((self.e0 - K) / self.ef)) / K)

class GKLocalDamageLinear: ### Developed now for numpy usage (NOT dolfin Ufl)
    """
    g(K) = (ku / (ku - ko)) * (1. - ko / K)
    """
    def __init__(self, ko=0.0001, ku=0.0125):
        self.ko = ko
        self.ku = ku
        self._fact = self.ku / (self.ku - self.ko) # as a helper value
    def g(self, K):
        condition = df.le(K, self.ko)
        return df.conditional(condition, 0, self._fact * (1.0 - self.ko / K))

class GKLocalDamagePerfect:
    def __init__(self, K0):
        self.K0 = K0
    def g(self, K):
        condition = df.le(K, self.K0)
        return df.conditional(condition, 0, 1 - self.K0 / K)

class MaterialGDM1D:
    def __init__(self, E, nu, gK, c, interaction_function=lambda D:1, epsilon_eq_num=None):
        self.E = E
        self.D = np.array([self.E])
        self.nu = nu
        self.gK = gK
        self.c = c # gradient damage constant (l^2 in Poh's 2017 paper)
        self.interaction_function = interaction_function # introduced as "g" in Poh's paper in 2017
        # self.fn_d_eps_eq_d_eps = lambda x: ufl.operators.sign(x)
    def sigma(self, u, K):
        return (1 - self.gK.g(K)) * self.E * ufl.grad(u)     
    def epsilon_eq(self, epsilon):  # equivalent strain
        return ufl.algebra.Abs(epsilon[0])
    def update_K(K_old, ebar):
        return ufl.operators.Max(K_old, ebar)

class BarGDM:
    def __init__(self):
        self.mesh = df.IntervalMesh(_res, 0., L)
        if damage_law == 'perfect':
            gK = GKLocalDamagePerfect(K0=e0)
        elif damage_law == 'linear':
            gK = GKLocalDamageLinear(ko=e0, ku=ef)
        elif damage_law == 'exp':
            gK = GKLocalDamageExponential(e0=e0, ef=ef, alpha=0.99)
        self.mat = MaterialGDM1D(E=E, nu=0, gK=gK, c=c)
        self.K0 = df.Expression('(x[0] >= x_from && x[0] <= x_to) ? k0 : 0.0', \
                             x_from=x_from, x_to=x_to, k0=K0_middle, degree=0)
        if red_factor==1 or x_from==x_to:
            self.sc_E = 1.0
        else:
            self.sc_E = df.Expression('(x[0] >= x_from && x[0] <= x_to) ? sc : 1.0', \
                             x_from=x_from, x_to=x_to, sc=red_factor, degree=0)
        self.f = df.Constant(0.0)
        md = {'quadrature_degree': integ_degree, 'quadrature_scheme': 'default'}
        self.dxm = df.dx(domain=self.mesh, metadata=md)
        df.parameters["form_compiler"]["quadrature_degree"] = integ_degree
        
        self._discretize_and_build_variationals()
        self._add_bcs()
        self._build_solver()
        
    def _discretize_and_build_variationals(self):
        elem_u = df.FiniteElement(el_family, self.mesh.ufl_cell(), shF_degree_u)
        elem_ebar = df.FiniteElement(el_family, self.mesh.ufl_cell(), shF_degree_ebar)
        elem_mix = df.MixedElement([elem_u, elem_ebar])
        self.i_mix = df.FunctionSpace(self.mesh, elem_mix)
        self.u_mix = df.Function(self.i_mix, name='Main fields')
        self.u_u, self.u_ebar = df.split(self.u_mix) # as indices
        self.v_mix  = df.TestFunction(self.i_mix)
        self.v_u, self.v_ebar = df.split(self.v_mix)
        # "u_K_current" is for internal variables
        e_K = df.FiniteElement(family="Quadrature", cell=self.mesh.ufl_cell(), \
                               degree=integ_degree, quad_scheme="default")
        self.i_K = df.FunctionSpace(self.mesh, e_K)
        self.u_K_current = df.interpolate(self.K0, self.i_K)
        
        ### Variationals (MIXED)
        # First Equation
        u_Kmax = MaterialGDM1D.update_K(self.u_K_current, self.u_ebar)
        self.sig_u = self.sc_E * self.mat.sigma(self.u_u, u_Kmax)
        eps_v = ufl.grad(self.v_u)
        a_u = df.inner(self.sig_u, eps_v) * self.dxm
        L_u = df.inner(self.f, self.v_u) * self.dxm
        F_u = a_u - L_u
        # Second Equation
        a_ebar = df.inner(self.u_ebar, self.v_ebar) * self.dxm
        interaction = self.mat.interaction_function(self.mat.gK.g(u_Kmax))
        a_ebar += self.mat.c * interaction * df.dot(ufl.grad(self.u_ebar), ufl.grad(self.v_ebar)) * self.dxm
        eps_eq = self.mat.epsilon_eq(ufl.grad(self.u_u))
        L_ebar = df.inner(eps_eq, self.v_ebar) * df.dx(self.mesh) # Here "eps_eq" is similar to an external force applying to the second field system
        F_ebar = a_ebar - L_ebar
        # Sum
        self.F = F_u + F_ebar
        
        ### Variationals (ONLY Ebar)
        # Second Equation
        self.i_u_alone = self.i_mix.sub(0).collapse()
        self.i_ebar_alone = self.i_mix.sub(1).collapse()
        self.u_u_alone = df.Function(self.i_u_alone, name='U_alone')
        self.u_ebar_alone = df.Function(self.i_ebar_alone, name='Ebar_alone')
        self.v_ebar_alone = df.TestFunction(self.i_ebar_alone)
        u_Kmax_alone = MaterialGDM1D.update_K(self.u_K_current, self.u_ebar_alone)
        a_ebar_alone = df.inner(self.u_ebar_alone, self.v_ebar_alone) * self.dxm
        interaction_alone = self.mat.interaction_function(self.mat.gK.g(u_Kmax_alone))
        a_ebar_alone += self.mat.c * interaction_alone * df.dot(ufl.grad(self.u_ebar_alone), ufl.grad(self.v_ebar_alone)) * self.dxm
        eps_eq_alone = self.mat.epsilon_eq(ufl.grad(self.u_u_alone))
        L_ebar_alone = df.inner(eps_eq_alone, self.v_ebar_alone) * self.dxm
        self.F_ebar = a_ebar_alone - L_ebar_alone
        
        self.u_dofs = self.i_mix.sub(0).dofmap().dofs()
        self.e_dofs = self.i_mix.sub(1).dofmap().dofs()
    
    def _add_bcs(self, _tol=1.0e-14):
        i_u = self.i_mix.sub(0)
        def left(x):
            return df.near(x[0], 0.0, _tol)
        bcl = df.DirichletBC(i_u, df.Constant(0.0), left)
        bcl_dofs = [k for k in bcl.get_boundary_values().keys()]
        def right(x):
            return df.near(x[0], L, _tol)
        self.u_right = df.Expression('u_max * t / T', degree=1, t=0, T=T, u_max=loading)
        bcr = df.DirichletBC(i_u, self.u_right, right)
        bcr_dofs = [k for k in bcr.get_boundary_values().keys()]
        self.bcs = [bcl, bcr]
        self.bcs_dofs = bcl_dofs + bcr_dofs
        
    def _build_solver(self):
        self.K_t_form = df.derivative(self.F, self.u_mix)
        problem = df.NonlinearVariationalProblem(self.F, self.u_mix, self.bcs, self.K_t_form)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
        solver.parameters["newton_solver"]["maximum_iterations"] = 10
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = solver_tol
        solver.parameters["newton_solver"]["relative_tolerance"] = solver_tol
        self.solver = solver
    
    def solve(self, t, allow_nonconvergence_error=False, max_iters=50):
        self.u_right.t = t
        (it, conv) = self.solver.solve()
        if conv:
            self._todo_after_convergence()
        return (it, conv)
    
    def _todo_after_convergence(self):
        k = self.u_K_current.vector().get_local()
        ebar = self.u_mix.split()[1]
        
        e = df.interpolate(ebar, self.i_K).vector().get_local()
        
        # df.parameters["form_compiler"]["representation"] = "quadrature"
        # e = df.project(ebar, self.i_K).vector().get_local()
        # df.parameters["form_compiler"]["representation"] = "uflacs"
        
        self.u_K_current.vector().set_local(np.maximum(k, e))
    
    def solve_over_time(self, checkpoints=[], pps=[]):
        ts = TimeStepper(self.solve, pps, solution_field=self.u_mix, _dt_max=1.0)
        dt0 = T / 30
        ts.adaptive(T, 0.0, dt=dt0, checkpoints=checkpoints)
    
    def solve_only_ebar(self, K0, uu):
        self.u_K_current.vector()[:] = K0
        self.u_u_alone.vector()[:] = uu ################## ?????? might be imprecise
        self.K_ebar = df.derivative(self.F_ebar, self.u_ebar_alone)
        problem = df.NonlinearVariationalProblem(self.F_ebar, self.u_ebar_alone, None, self.K_ebar)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
        solver.parameters["newton_solver"]["maximum_iterations"] = 10
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = solver_tol
        solver.parameters["newton_solver"]["relative_tolerance"] = solver_tol
        self.solver_only_ebar = solver
        (it, conv) = self.solver_only_ebar.solve()
        assert conv
        k = self.u_K_current.vector().get_local()
        e = df.interpolate(self.u_ebar_alone, self.i_K).vector().get_local()
            # df.parameters["form_compiler"]["representation"] = "quadrature"
            # e = df.project(ebar, self.i_K).vector().get_local()
            # df.parameters["form_compiler"]["representation"] = "uflacs"
        self.u_K_current.vector().set_local(np.maximum(k, e))
        

class PostProcessGDM:
    def __init__(self, model, points=None, reaction_dofs=None, log_residual_vector=False \
                 , write_files=True, _eval=True, _name='pp_gdm', out_path='./'):
        self.model = model
        self.points = points or self.model.mesh.coordinates()
        self.reaction_dofs = reaction_dofs
        self.log_residual_vector = log_residual_vector
        self.write_files = write_files
        self._eval = _eval
        self.full_name = out_path + _name
        self.i_DG0 = df.FunctionSpace(self.model.mesh, "DG", 0)
        self.D = df.Function(self.i_DG0, name='Damage')
        self.K = df.Function(self.i_DG0, name='Kappa')
        self.checkpoints = [] # will be overwritten as soon as a time-stepper incremental solution is executed.
        self._reset()
    def _reset(self):
        self.ts = []
        self.reaction_forces = [] # Every entry is for a certain time ("t") and is a list of values over the given reaction_dofs
        self.residual_norms = [] # # Every entry is for a certain time ("t") and is the residual norm at all FREE DOFs (excluding Dirichlet BCs)
        if self.write_files:
            self.remove_files()
            self.xdmf_checked = df.XDMFFile(self.full_name + '_checked.xdmf')
                # :including the underlying FEniCS functions (used for functions evaluations)
            self.xdmf = df.XDMFFile(self.full_name + '.xdmf')
                # :for visualizations in Paraview
            self.xdmf.parameters["functions_share_mesh"] = True
            self.xdmf.parameters["flush_output"] = True
        if self._eval:
            self.all = {'u':[], 'ebar':[], 'Kappa':[], 'Damage':[]}
            self.checked = {'u':[], 'ebar':[], 'Kappa':[], 'Damage':[]}
            ## with capital first letter implies over function space directly, i.e. the vector() of functions and without any evaluation
            self.All = {'u':[], 'ebar':[], 'Kappa':[]}
            self.Checked = {'u':[], 'ebar':[], 'Kappa':[]}
    def __call__(self, t, logger=None):
        if t==0.0: # initiation
            self._reset()
        self.ts.append(t)
        reaction_force, b_norm = compute_residual(F=self.model.F, bcs_dofs=self.model.bcs_dofs, \
                                             reaction_dofs=[self.reaction_dofs], logger=logger, write_residual_vector=self.log_residual_vector)
        if reaction_force is not None:
            self.reaction_forces.append(reaction_force[0])
        self.residual_norms.append(b_norm)
        if self.write_files or self._eval:
            u_plot, ebar_plot = self.model.u_mix.split(deepcopy=True)
            u_plot.rename('Displacements', 'Displacements')
            ebar_plot.rename('Ebar', 'Ebar')
            df.project(v=self.model.u_K_current, V=self.i_DG0, function=self.K) # projection
            D_ufl = self.model.mat.gK.g(self.model.u_K_current)
            df.project(v=D_ufl, V=self.i_DG0, function=self.D) # projection
        if self.write_files:
            self.xdmf.write(u_plot, t)
            self.xdmf.write(ebar_plot, t)
            self.xdmf.write(self.K, t)
            self.xdmf.write(self.D, t)
        if self._eval:
            u_vals = np.array([u_plot(p) for p in self.points])
            ebar_vals = np.array([ebar_plot(p) for p in self.points])
            K_vals = self.K.vector().get_local()
            D_vals = self.D.vector().get_local()
            self.all['u'].append(u_vals)
            self.all['ebar'].append(ebar_vals)
            self.all['Kappa'].append(K_vals)
            self.all['Damage'].append(D_vals)
            
            u_Vals = self.model.u_mix.vector().get_local()[self.model.u_dofs]
            ebar_Vals = self.model.u_mix.vector().get_local()[self.model.e_dofs]
            K_Vals = self.model.u_K_current.vector().get_local()
            self.All['u'].append(u_Vals)
            self.All['ebar'].append(ebar_Vals)
            self.All['Kappa'].append(K_Vals)
        
        if self.write_files or self._eval:
            for tt in self.checkpoints:
                if abs(t - tt) < 1e-9:
                    if self.write_files:
                        self.xdmf_checked.write_checkpoint(u_plot, 'u', t, append=True)
                        self.xdmf_checked.write_checkpoint(ebar_plot, 'ebar', t, append=True)
                        self.xdmf_checked.write_checkpoint(self.K, 'Kappa', t, append=True)
                        self.xdmf_checked.write_checkpoint(self.D, 'Damage', t, append=True)
                    if self._eval:
                        self.checked['u'].append(u_vals)
                        self.checked['ebar'].append(ebar_vals)
                        self.checked['Kappa'].append(K_vals)
                        self.checked['Damage'].append(D_vals)
                        self.Checked['u'].append(u_Vals)
                        self.Checked['ebar'].append(ebar_Vals)
                        self.Checked['Kappa'].append(K_Vals)
    
    def plot_residual_at_free_DOFs(self, tit='Residuals at free DOFs', full_file_name=None, factor=1, marker='.'):
        fig1 = plt.figure()
        plt.plot(self.ts, [factor * res for res in self.residual_norms], marker=marker)
        plt.title(tit)
        plt.xlabel('t')
        plt.ylabel('Residual')
        if full_file_name is not None:
            plt.savefig(full_file_name)
        plt.show()
    def plot_reaction_forces(self, tit='Reaction Forces', dof='sum', full_file_name=None, factor=1, marker='.', sz=14):
        fig1 = plt.figure()
        if dof=='sum':
            if type(self.reaction_forces[0])==list or type(self.reaction_forces[0])==np.ndarray:
                f_dof = [factor * sum(f) for f in self.reaction_forces]
            else:
                f_dof = [factor * f for f in self.reaction_forces]
        else:
            f_dof = [factor * f[dof] for f in self.reaction_forces]
        plt.plot(self.ts, f_dof, marker=marker)
        plt.title(tit, fontsize=sz)
        plt.xlabel('t', fontsize=sz)
        plt.ylabel('f', fontsize=sz)
        plt.xticks(fontsize=sz)
        plt.yticks(fontsize=sz)
        
        if full_file_name is not None:
            plt.savefig(full_file_name)
        plt.show()
        return f_dof
    def close_files(self):
        if self.write_files:
            self.xdmf.close()
            self.xdmf_checked.close()
    def remove_files(self):
        import os
        if os.path.exists(self.full_name + '.xdmf'):
            os.remove(self.full_name + '.xdmf')
            os.remove(self.full_name + '.h5')
        if os.path.exists(self.full_name + '_checked.xdmf'):
            os.remove(self.full_name + '_checked.xdmf')
            os.remove(self.full_name + '_checked.h5')

def main_study_bar():
    #### MODEL
    model = BarGDM()
    
    #### FORWARD SOLUTION
    checkpoints=np.linspace(0, 1, 5)
    pp = PostProcessGDM(model, reaction_dofs=[model.bcs_dofs[1]])
    pps = [pp]
    model.solve_over_time(checkpoints=checkpoints, pps=pps)
    df.plot(model.u_mix.sub(0)); plt.title('Final displacements'); plt.show()
    pp.plot_reaction_forces()
    pp.plot_residual_at_free_DOFs()
    dg0_cs = pp.i_DG0.tabulate_dof_coordinates().flatten() # coordinates of DG0 space
    ebar_cs = pp.points.flatten() # coordinates over which Ebar has been checked
    for i in range(len(checkpoints)):
        plt.figure()
        plt.plot(dg0_cs, pp.checked['Kappa'][i], label='Kappa')
        plt.plot(ebar_cs, pp.checked['ebar'][i], label='ebar')
        plt.title('Damage law: ' + damage_law + f", Load-step {i}")
        plt.legend(); plt.show()
    
    #### IMPOSITION (of Displacements to the model)
    K0 = 0 # for the first load-step the initial Kappa is zero
    for i in range(len(checkpoints)):
        K_updated = pp.Checked['Kappa'][i] # from mixed solution
        ## Ebar solution only
        uu = pp.Checked['u'][i]
        model.solve_only_ebar(K0=K0, uu=uu)
        K_updated_alone = model.u_K_current.vector().get_local()
        plt.figure()
        plt.plot(K_updated, label='Mixed solution')
        plt.plot(K_updated_alone, label='Ebar solution only')
        plt.plot(K_updated - K_updated_alone, label='Error')
        plt.title('Kappa (Damage law: ' + damage_law + f") Load-step {i}")
        plt.legend(); plt.show()
        K0 = K_updated # for next load-step
        
        # update corresponding fields of mix problem and compute residuals at free DOFs
             ################## ?????? might be imprecise
        model.u_mix.vector()[model.u_dofs] = uu
        model.u_mix.vector()[model.e_dofs] = model.u_ebar_alone.vector().get_local()
        res = df.assemble(model.F).get_local()[0:]
        plt.figure()
        plt.plot(res, marker='.', linestyle='')
        plt.title('Residuals after imposition of displacements' + f"\nLoad-step {i}")
        plt.show()
    
    return model, pps

if __name__ == "__main__":
    df.set_log_level(30)
    
    model, pps = main_study_bar()
    
    df.set_log_level(20)
