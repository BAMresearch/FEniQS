from feniQS.fenics_helpers.fenics_functions import *

class MyStructure:
    def __init__(self):
        self.lx = self.ly = 1.
        self.mesh = df.UnitSquareMesh(10, 10)
    
    def get_BCs_fixed_bottom(self, iu):
        bcs_fixed = {}
        tol = self.mesh.rmax() / 1000.
        
        def bot(x, on_boundary):
            return df.near(x[1], 0., tol)
        
        bc = df.DirichletBC(iu.sub(0), df.Constant(0.), bot)
        bc_dofs = [key for key in bc.get_boundary_values().keys()]
        bcs_fixed['bot_x'] = {'bc': bc, 'dofs': bc_dofs}

        bc = df.DirichletBC(iu.sub(1), df.Constant(0.), bot)
        bc_dofs = [key for key in bc.get_boundary_values().keys()]
        bcs_fixed['bot_y'] = {'bc': bc, 'dofs': bc_dofs}

        return bcs_fixed

    def get_BCs_and_loading(self, iu, top_load, top_load_type='f'):
        bcs = self.get_BCs_fixed_bottom(iu)

        tol = self.mesh.rmax() / 1000.
        def top(x, on_boundary):
            return df.near(x[1], self.ly, tol)
        
        bcs_Neumann = dict()

        # l_y = df.Expression('t * ( sin(x[0]) + cos(x[1])*cos(x[1]) )', degree=1, t=1)
        l_y = df.Expression('t * l / T', degree=0, t=1., T=1., l=top_load)
        loadings = {'top_load': l_y}
        
        if top_load_type.lower()=='u': # Dirichlet BCs
            bc = df.DirichletBC(iu.sub(1), l_y, top)
            bc_dofs = [key for key in bc.get_boundary_values().keys()]
            bcs['top_y'] = {'bc': bc, 'dofs': bc_dofs}
        elif top_load_type.lower()=='f':
            f_0 = df.Constant(0.0)
            t = df.as_tensor((f_0, l_y))
            dom_top = df.AutoSubDomain(top)
            mf = df.MeshFunction('size_t', self.mesh, 1) # 1 goes to edges
            mf.set_all(0)
            dom_top.mark(mf, 1)
            ds = df.Measure('ds', domain=self.mesh, subdomain_data=mf)
            bcs_Neumann['top_y'] = {'t': t, 'ds': ds(1)}
        else:
            raise ValueError('Invalid loading type.')

        return loadings, bcs, bcs_Neumann

class MyElasticMaterial():
    def __init__(self, E, nu, constraint='PLANE_STRESS'):
        self.E = E
        self.nu = nu
        self.constraint = constraint
        self.dim = 2
        _fact = 1 # Voigt notation
        self.ss_dim = 3
        self.D = (self.E / (1 - self.nu ** 2)) * np.array([ [1, self.nu, 0], [self.nu, 1, 0], [0, 0, _fact * 0.5 * (1-self.nu) ] ])

        self.lamda = (self.E * self.nu / (1 + self.nu)) / (1 - 2*self.nu)
        self.mu = self.E / (2 * (1 + self.nu))
        if self.constraint=='PLANE_STRESS':
            self.lamda = 2 * self.mu * self.lamda / (self.lamda + 2 * self.mu)
        
    # def sigma(self, u):
    #     eps_u = df.sym(df.grad(u))
    #     return self.lamda * df.tr(eps_u) * df.Identity(2) + 2 * self.mu * eps_u

def my_eps_vector(u):
    e = df.sym(df.grad(u))
    _fact = 2 # Voigt notation
    return df.as_vector([e[0, 0], e[1, 1], _fact * e[0, 1]])

class MyElasticProblem(df.NonlinearProblem):
    def __init__(self, struct, mat):
        df.NonlinearProblem.__init__(self)
        self.struct = struct
        self.mesh = self.struct.mesh
        self.mat = mat
        self.shF_degree_u = 2
        self.integ_degree = 2

        md = {'quadrature_degree': self.integ_degree,
              'quadrature_scheme': 'default'}
        self.dxm = df.dx(domain = self.mesh, metadata = md)
        df.parameters["form_compiler"]["quadrature_degree"] = self.integ_degree
        df.parameters["form_compiler"]["representation"] = "quadrature"
        import warnings
        from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
        warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

        elem_u = df.VectorElement('CG', self.mesh.ufl_cell(), self.shF_degree_u, dim=self.mat.dim)
        self.i_u = df.FunctionSpace(self.mesh, elem_u)
        # Define functions
        self.u_u = df.Function(self.i_u, name="Displacement")
        self.v = df.TrialFunction(self.i_u)
        self.v_u = df.TestFunction(self.i_u)
        self.nodal_external_forces = df.Function(self.i_u, name="Nodal external forces")
        self.nodal_external_forces.vector()[:] = 0.
        
        ### Quadrature spaces / functions
        # Gauss-points
        elem_scalar_gauss = df.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree \
                                             , quad_scheme="default")
        self.i_q = df.FunctionSpace(self.mesh, elem_scalar_gauss)
        self.ngauss = self.i_q.dim() # get total number of gauss points
        # For sigma and strain
        elem_ss = df.VectorElement("Quadrature", self.mesh.ufl_cell() \
                                   , degree=self.integ_degree, dim=self.mat.ss_dim, quad_scheme="default")
        self.i_ss = df.FunctionSpace(self.mesh, elem_ss)
        # For tangent matrix
        elem_tensor = df.TensorElement("Quadrature", self.mesh.ufl_cell() \
                                       , degree=self.integ_degree, shape=(self.mat.ss_dim, self.mat.ss_dim) \
                                       , quad_scheme="default")
        i_tensor = df.FunctionSpace(self.mesh, elem_tensor)
        
        # Define functions based on Quadrature spaces
        self.sig = df.Function(self.i_ss, name="Stress")
        self.eps = df.Function(self.i_ss, name="Strain")
        self.projector_eps = LocalProjector(my_eps_vector(self.u_u), self.i_ss, self.dxm)
        
        # Define and initiate tangent operator (elasto-plastic stiffness matrix)
        self.Ct = df.Function(i_tensor, name="Tangent operator")
        Ct_num = np.tile(self.mat.D.flatten(), self.ngauss).reshape((self.ngauss, self.mat.ss_dim**2)) # set the elastic stiffness at all Gauss-points
        self.Ct.vector().set_local(Ct_num.flatten()) # assign the helper "Ct_num" to "Ct"

        self.K_t = df.inner(my_eps_vector(self.v), df.dot(self.Ct, my_eps_vector(self.v_u))) * self.dxm
        self.F_int = df.inner(my_eps_vector(self.v_u), self.sig) * self.dxm
        self.F_ext = 0.

    def build_solver(self, bcs, bcs_Neumann):
        self.bcs = bcs
        self.bcs_Neumann = bcs_Neumann
        for bc_N in bcs_Neumann.values():
            self.F_ext += df.dot(bc_N['t'], self.v_u) * bc_N['ds']
        self.projector_eps = LocalProjector(my_eps_vector(self.u_u), self.i_ss, self.dxm)
        self.assembler = df.SystemAssembler(self.K_t, self.F_int - self.F_ext, bcs)
        nln_so = get_my_nonlinear_solver_options()
        self.solver = get_my_solver(nln_so=nln_so)
    
    def F(self, b, x):
        # project the corresponding strain to quadrature space
        self.projector_eps(self.eps)
        # compute correct stress and Ct based on the updated strain "self.eps"
        Cts = self.Ct.vector().get_local().reshape((-1, self.mat.ss_dim, self.mat.ss_dim))
        strains = self.eps.vector().get_local().reshape((-1, self.mat.ss_dim))
        self.sig.vector()[:] = np.einsum('mij,mj->mi', Cts, strains).flatten()
        # update the solver's residual
        self.assembler.assemble(b, x)
        b[:] += self.nodal_external_forces.vector().get_local()[:]

    def J(self, A, x):
        # update the solver's tangent operator
        self.assembler.assemble(A)

    def solve(self, t=0):
        u = self.u_u
        _it, conv = self.solver.solve(self, u.vector())
        if conv:
            print(f"    The time step t={t} converged after {_it} iteration(s).")
        else:
            print(f"    The time step t={t} did not converge.")
        return (_it, conv)
    
    def compute_strain_energy(self):
        strain_energy_density = 0.5 * df.inner(self.sig, self.eps) * self.dxm
        return df.assemble(strain_energy_density)

def get_my_nonlinear_solver_options():
    return {
    'type': "newton",
    'max_iters': 14,
    'allow_nonconvergence_error': False,
    'tol_abs': 1e-10,
    'tol_rel': 1e-10,
    }

def get_my_solver(nln_so):
    solver = df.NewtonSolver()
    solver.parameters["absolute_tolerance"] = nln_so['tol_abs']
    solver.parameters["relative_tolerance"] = nln_so['tol_rel']
    solver.parameters["error_on_nonconvergence"] = nln_so['allow_nonconvergence_error']
    solver.parameters["maximum_iterations"] = nln_so['max_iters']
    return solver

def compute_strain_energy(fen, _info=''):
    K_t = df.assemble(fen.K_t).array()
    u_vec = fen.u_u.vector().get_local()
    E1 = 0.5 * np.einsum('m,mn,n->', u_vec, K_t, u_vec)

    f_int = df.assemble(fen.F_int).get_local()
    E2 = 0.5 * np.einsum('n,n->', u_vec, f_int)

    E3 = fen.compute_strain_energy()

    if _info!='':
        _info = f" ({_info})"
    _end = max((34 - len(_info)), 0) * '-'
    _msg = f"\033[33mStrain Energy{_info}\033[0m {_end}"
    _msg += f"\n\t1) (U^T).K.U               = \033[32m{E1:.5e}\033[0m"
    _msg += f"\n\t2) (U^T).F_int             = \033[32m{E2:.5e}\033[0m"
    _msg += f"\n\t3) Integral(Stress:Strain) = \033[32m{E3:.5e}\033[0m"
    _msg += f"\n\t                     E1-E2 = \033[31m{(E1-E2):.1e}\033[0m"
    _msg += f"\n\t                     E1-E3 = \033[31m{(E1-E3):.1e}\033[0m"
    _msg += f"\n\t                     E2-E3 = \033[31m{(E2-E3):.1e}\033[0m"
    _msg += f"\n\033[0m------------------------------------------------"
    print(_msg)

    return (E1, E2, E3)

if __name__=='__main__':
    df.set_log_level(30)
    
    ###################################################################
        ### Setup (base)
    ###################################################################
    struct = MyStructure()
    mat = MyElasticMaterial(E=200e3, nu=0.3)
    
    ###################################################################
        ### Reference problem 'fen0' (as admissible stresses)
    ###################################################################
    top_load0 = -1000.; top_load_type0 = 'f'
    # top_load0 = -0.005; top_load_type0 = 'u'
    fen0 = MyElasticProblem(struct=struct, mat=mat)
    loadings0, bcs0_dict, bcs0_Neumann = struct.get_BCs_and_loading(iu=fen0.i_u \
                        , top_load=top_load0, top_load_type=top_load_type0)
    bcs0 = [bc['bc'] for bc in bcs0_dict.values()]
    fen0.build_solver(bcs=bcs0, bcs_Neumann=bcs0_Neumann)
    fen0.solve()
        # Admissible stresses and forces (in equilibrium with external forces)
    sig_ad = copy.deepcopy(fen0.sig.vector().get_local())
    fs_ad = df.assemble(fen0.F_int).get_local()
        # Strain energy
    E0s = compute_strain_energy(fen0, 'Reference (admissible)')
        # Displacements everywhere except bottom edge
    cs = fen0.mesh.coordinates()
    tol = fen0.mesh.rmin()/1000.
    cs_selected = cs[[tol<y for y in cs[:,1]],:] # Exclude bottom edge
    us_selected = np.array([fen0.u_u(c) for c in cs_selected])
    
    ###################################################################
        ### Arbitrary problem 'fen' (e.g. by imposing some biased and noisy Disps.)
    ###################################################################
    fen = MyElasticProblem(struct=struct, mat=mat)
        # Build many_bcs for some noisy displacements (except at bottom edge)
    scale_us = 1.2
    noise_level = 0.2
    np.random.seed(1983)
    us_noise = np.random.normal(0., noise_level * np.std(us_selected), us_selected.shape)
    us_selected_noisy = scale_us * us_selected + us_noise
    many_bcs = ManyBCs(V=fen.i_u, v_bc=fen.i_u, points=cs_selected \
                       , sub_ids=[], vals=us_selected_noisy)
        # We also need only fixed Dirichlet BCs at bottom edge
    bcs_fixed_dict = struct.get_BCs_fixed_bottom(iu=fen.i_u)
    bcs_fixed = [bc['bc'] for bc in bcs_fixed_dict.values()]
        # Solve
    fen.build_solver(bcs=bcs_fixed + [many_bcs.bc], bcs_Neumann=dict())
    fen.solve()
        # Strain energy
    Es = compute_strain_energy(fen, 'By imposing disps.')
    f_int = df.assemble(fen.F_int).get_local()
    sig = copy.deepcopy(fen.sig.vector().get_local())
    eps = copy.deepcopy(fen.eps.vector().get_local())

    ###################################################################
        ### CEG error (of strain energy norm) computed from:
            # d_sigma_CEG        = sig - sig_ad
            # eps_of_d_sigma_CEG = (C^-1).d_sigma_CEG
        # where both 'sig' and 'C' are from the 'fen' problem above.
    ###################################################################
    d_sigma_CEG = sig - sig_ad
        # We use the mesh/problem of 'fen' to compute err_CEG (we have already back-up of fen.sig and fen.eps)
    fen.sig.vector()[:] = d_sigma_CEG[:]
    eps_of_d_sigma_CEG = np.zeros_like(eps)
    C_matrices = fen.Ct.vector().get_local().reshape((fen.i_q.dim(),3,3))
    for i, C in enumerate(C_matrices):
        C_inv = np.linalg.inv(C)
        eps_of_d_sigma_CEG_i = C_inv @ d_sigma_CEG[i*3:(i+1)*3]
        eps_of_d_sigma_CEG[i*3:(i+1)*3] = eps_of_d_sigma_CEG_i[:]
    fen.eps.vector()[:] = eps_of_d_sigma_CEG
    plt.figure()
    plt.plot(eps, label='Epsilon computed after imposing disps.')
    plt.plot(fen.eps.vector(), label='Epsilon = (C^-1).d_sigma_CEG (used for CEG error)')
    plt.legend()
    plt.show()
    E_CEG = fen.compute_strain_energy()
    print(f"\n\033[33mStrain energy (CEG)\033[0m\n\tIntegral(Stress:Strain) = \033[32m{E_CEG:.5e}\033[0m")
        # Reset fen.sig and fen.eps from their back-up values
    fen.sig.vector()[:] = sig[:]
    fen.eps.vector()[:] = eps[:]

    ###################################################################
        ### Build/solve the FFF problem; a problem with:
            # external forces being FEMU-F error,
            # constitutive matrix being the same as used for CEG error.
        ### Compute the strain energy of FFF problem.
    ###################################################################
    fen_fff = MyElasticProblem(struct=struct, mat=mat) # Has no BCs
        # Build the FEMU-F error (gap btw. admissible and internal forces)
    fs_femu_f = fs_ad - f_int
        # FEMU-F forces should be applied at everywhere except fixed BCs (otherwise the solution would diverge)
    dofs_bot = bcs_fixed_dict['bot_x']['dofs'] \
             + bcs_fixed_dict['bot_y']['dofs']
    dofs_except_bot = [i for i in range(fen_fff.i_u.dim()) if i not in dofs_bot]
    plt.figure(); plt.plot(fs_femu_f[dofs_except_bot])
    plt.title(f"FEMU-F error forces (except at fixed BCs)"); plt.show()
        # Set nodal external forces (everywhere except fixed BCs)
    fen_fff.nodal_external_forces.vector()[dofs_except_bot] = fs_femu_f[dofs_except_bot]
        # We also need only fixed Dirichlet BCs at bottom edge
    bcs_fixed_dict = struct.get_BCs_fixed_bottom(iu=fen_fff.i_u)
    bcs_fixed = [bc['bc'] for bc in bcs_fixed_dict.values()]
        # Solve
    fen_fff.build_solver(bcs=bcs_fixed, bcs_Neumann=dict())
    fen_fff.solve()
        # Strain energy
    Efffs = compute_strain_energy(fen_fff, 'FEMU-F error as nodal forces')
    
    ###################################################################
        ### Comparisons & Plots
    ###################################################################
        # Compare E_CEG and Efffs
    err_r = abs((Efffs[0] - E_CEG) / E_CEG)
    print(f"\033[33mRelative error btw. CEG error and FFF energy\n\t\033[31m{err_r:.1e}\033[0m.")
        # Compare stresses and strains of CEG and FFF
    err_sig = fen_fff.sig.vector().get_local() - d_sigma_CEG
    err_eps = fen_fff.eps.vector().get_local() - eps_of_d_sigma_CEG
    plt.figure(); plt.plot(err_sig); plt.title('Error of Stress btw. CEG and FFF'); plt.show()
    plt.figure(); plt.plot(err_eps); plt.title('Error of Strain btw. CEG and FFF'); plt.show()
        # Investigate balance of energy
    energy_balance = Es[0] - (E0s[0] + Efffs[0])
    _msg = f"\033[33mEnergy balance\033[0m = E - E_admissible - E_FFF =\n\t\033[31m{energy_balance:.2e}\033[0m."
    _msg += f"\n\tThis can be nonzero; e.g. if the admissible and FFF problems have different displacement-responses and/or Neumann BCs!"
    print(_msg)
    plt.figure(); df.plot(fen0.u_u); plt.title(f"U (reference (admissible) problem)"); plt.show()
    plt.figure(); df.plot(fen.u_u); plt.title(f"U (arbitrary problem)"); plt.show()
    plt.figure(); df.plot(fen_fff.u_u); plt.title(f"U (FFF problem)"); plt.show()


    df.set_log_level(20)