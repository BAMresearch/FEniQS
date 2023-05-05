from feniQS.general.parameters import *
from feniQS.problem.problem import *

pth_problem_elastic_homogenized = CollectPaths('pth_problem_elastic_homogenized.py')
pth_problem_elastic_homogenized.add_script(pth_parameters)
pth_problem_elastic_homogenized.add_script(pth_problem)

class ParsHomogenization(ParsBase):
    def __init__(self, pars0=None, **kwargs):
        """
        pars0 normally represents the material parameters.
        """
        ParsBase.__init__(self, pars0)
        if len(kwargs)==0: # Default values
            self.hom_type = None # either 'Dirichlet' or 'Periodic'
            print(f"WARNING: Homogenization type isn't yet specified.")
        else: # Get from a dictionary
            ParsBase.__init__(self, **kwargs)

def build_homogenization_problem(mesh, pars_hom, **kwargs):
    assert isinstance(pars_hom, ParsHomogenization)
    if pars_hom.mat_type.lower() != 'elastic':
        raise ValueError(f"Homogenization problem is only implemented for an elastic material model.")
    if pars_hom.hom_type is None:
        raise ValueError(f"The type pf homogenization is not specified. Set it to either 'Dirichlet' or 'Periodic'.")
    try:
        RVE_volume = kwargs['RVE_volume']
    except KeyError:
        raise KeyError(f"For a homogenization problem, the 'RVE_volume' must be given as input.")
    if 'dirichlet' in pars_hom.hom_type.lower():
        try:
            RVE_boundaries = kwargs['RVE_boundaries']
        except KeyError:
            raise KeyError(f"For Dirichlet homogenization, the 'RVE_boundaries' must be given as input.")
        mat = ElasticConstitutive(E=pars_hom.E+pars_hom.E_min, nu=pars_hom.nu, constraint=pars_hom.constraint)
        hom_problem = HomogenizationElasticDirichlet(mat=mat, mesh=mesh, fen_config=pars_hom, RVE_volume=RVE_volume)
        hom_problem.build_variational_functionals(f=pars_hom.f, integ_degree=pars_hom.integ_degree)
        hom_problem.build_DirichletBC(RVE_boundaries)
    elif 'periodic' in pars_hom.hom_type.lower():
        raise NotImplementedError(f"Periodic homogenization isn't yet implemented.")
    else:
        raise ValueError(f"The type pf homogenization is not recognized. Possible options are 'Dirichlet' and 'Periodic'.")
    
    return hom_problem

def estimate_RVE_volume(mesh):
    """
    Given a mesh and fitting its domain into a rectangle/box (as for 2D/3D geometry),
    this method returns:
        RVE_volume: the area/volume of that rectangle/box (as an estimation for the RVE's volume)
    """
    geo_dim = mesh.geometric_dimension()
    cs = mesh.coordinates()
    RVE_volume = 1.
    for i in range(geo_dim):
        a1, a2 = np.min(cs[:,i]), np.max(cs[:,i])
        RVE_volume *= abs(a1-a2)
    return RVE_volume

def get_RVE_boundaries(mesh):
    """
    Given a mesh and fitting its domain into a rectangle/box (as for 2D/3D geometry),
    this method returns:
        RVE_boundaries: a callable representing the exterior boarders of that rectangle/box
        , which is then used for setting a Dirichlet BC related to the boundaries of RVE.
    """
    geo_dim = mesh.geometric_dimension()
    cs = mesh.coordinates()
    X12 = [] # min and max of coordinates per each (possible) direction
    for i in range(geo_dim):
        X12.append([np.min(cs[:,i]), np.max(cs[:,i])])
    tol = mesh.rmin() / 1000.
    def RVE_boundaries(x, on_boundary):
        b = False
        i = 0
        while (not b) and (i<geo_dim):
            b = b or df.near(x[i], X12[i][0], tol) or df.near(x[i], X12[i][1], tol)
            i += 1
        return b
    return RVE_boundaries

def get_macro_strain_voigt(dim, case, strain_scalar):
    """
    Returns the macroscopic strain field for the given dimension and load-case in vectorized Voigt notion.
    """
    if dim == 1:
        raise NotImplementedError()
    elif dim == 2:
        assert case < 3
        Eps_Voigt = np.zeros((3,))
        Eps_Voigt[case] = strain_scalar
        Eps_Voigt[2] *= 0.5
    elif dim == 3:
        assert case < 6
        Eps_Voigt = np.zeros((6))
        Eps_Voigt[case] = strain_scalar
        for i in [3, 4, 5]: # due to Voigt notation
            Eps_Voigt[i] *= 0.5
    return Eps_Voigt

class HomogenizationElastic(FenicsElastic):
    def __init__(self, mat, mesh, fen_config, RVE_volume, dep_dim=None):
        FenicsElastic.__init__(self, mat=mat, mesh=mesh, fen_config=fen_config, dep_dim=dep_dim)
        
        if self.dep_dim == 1:
            raise NotADirectoryError(f"For 1-D problem is not implemented.")
        elif self.dep_dim == 2:
            self.cases = ["Exx", "Eyy", "Exy"]
            self.size_Chom = 3
        elif self.dep_dim == 3:
            self.cases = ["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"]
            self.size_Chom = 6            
        self.Chom = np.zeros((self.size_Chom, self.size_Chom)) # Homogenized elastic tensor
        self.Sigma = np.zeros((self.size_Chom,)) # Stress vecotr
        self.RVE_volume = RVE_volume # The volume of a whole RVE that is homogenized (as continuous domain).
        
    def update_case(self, j):
        self.case = self.cases[j]
    
    def get_sigma(self):
        """
        Depending on the type of homogenization (either Dirichlet or Periodic), specialized method will be overwritten.
        """
        raise NotImplementedError(f"Should be overwritten.")
    
    def calc_Chom_row(self, j): #(stress?, size_Chom, dim, dx, vol, strain_scalar)
        stress = self.get_sigma()
        voigt_stress = sigma_vector(sigma=stress, constraint=self.mat.constraint, ss_vector='voigt')
        if len(voigt_stress)==4:
            voigt_stress = [voigt_stress[0], voigt_stress[1], voigt_stress[3]]
        for k in range(self.size_Chom):
            self.Sigma[k] = df.assemble(voigt_stress[k] * self.dxm) / self.RVE_volume
        self.Chom[j, :] = self.Sigma / self.strain_scalar
        
    def do_postprocessing(self, j):
        self.calc_Chom_row(j)
        # super().do_postprocessing()
        # setattr(self, "vM_" + self.case, self.vM)
        # setattr(self, "vM_max_" + self.case, max(self.vM.vector().get_local()))
        
    def get_compliance_matrix(self):
        self.Shom = np.linalg.inv(self.Chom)
        self.E_hom = 1/self.Shom[0, 0]
        print("E_hom = ", self.E_hom)
        self.nu_hom = self.Shom[0, 1] * self.E_hom
        print("nu_hom = ", self.nu_hom)
        if self.dep_dim == 3:
            self.G_hom = 1/self.Shom[4, 4]
        elif self.dep_dim == 2:
            self.G_hom = 1/self.Shom[2, 2]
        else:
            raise NotImplementedError()
        print("G_hom = ", self.G_hom)
        
    def run_hom(self):
        print(f"\nSolving homogenization problem of {self.__class__.__name__} in {self.dep_dim}-D.")
        
        # if self.pars.rel_dens_trgt != None:
        #     self.pars = mesh_module.optimize_par_for_rel_density(self.pars)
        # self.mesh, self.rel_dens = mesh_module.create_mesh(self.pars) # better: self.create_mesh() ??
        # print(f"The {self.pars.shape} shape created with {self.pars.mesher}",
        #       f" in {self.pars.dim}D has a relative density of",
        #       f"{round(self.rel_dens, ndigits = 20)}.")
        
        for j in range(self.size_Chom):
            self.update_case(j)
            self.solve(t=0.)
            # PP
            self.do_postprocessing(j)
            # self.plot_case()
            # self.export_as_xdmf()
            # prob.save_as_attribute()
        
        self.get_compliance_matrix()
        # self.E_hom, self.nu_hom = calc_homogenized(self.Chom, self.pars.dim)
        # return prob

class HomogenizationElasticDirichlet(HomogenizationElastic):
    def __init__(self, mat, mesh, fen_config, RVE_volume, dep_dim=None, strain_scalar=1.0):
        HomogenizationElastic.__init__(self, mat=mat, mesh=mesh, fen_config=fen_config, RVE_volume=RVE_volume, dep_dim=dep_dim)
        self.strain_scalar = strain_scalar
        self._initiate_macro_strain_voigt()
        self._build_Dirichlet_u_expr()
    
    def _initiate_macro_strain_voigt(self):
        if self.dep_dim == 1:
            raise NotImplementedError()
        elif self.dep_dim == 2:
            self._Eps_Voigt = np.zeros((3,))
        elif self.dep_dim == 3:
            self._Eps_Voigt = np.zeros((6,))
        self.m_strain = df.Constant(self._Eps_Voigt)
    
    def _build_Dirichlet_u_expr(self, degree=1):
        """
        Build self.dr_expr, representing the displacement field from a certain macro-strain field that is the 's' attribute of that expression.
        """
        # u_str:
            # A tuple of strings to be used inside of an expression
            # , which represents displacement field according to coordinates (x) and a Voigt-vectorized strain field (s).
        if self.dep_dim == 1:
            raise NotImplementedError()
        elif self.dep_dim == 2:
            strain_x = "s[0] * x[0] + s[2] * x[1]"
            strain_y = "s[2] * x[0] + s[1] * x[1]"
            u_str = (strain_x, strain_y)
        elif self.dep_dim == 3:
            strain_x = "s[0] * x[0] + s[5] * x[1] + s[4] * x[2]"
            strain_y = "s[5] * x[0] + s[1] * x[1] + s[3] * x[2]"
            strain_z = "s[4] * x[0] + s[3] * x[1] + s[2] * x[2]"
            u_str = (strain_x, strain_y, strain_z)
        self.dr_expr = df.Expression(u_str, s=self.m_strain, degree=degree)

    def build_DirichletBC(self, RVE_boundaries):
        hom_bc = df.DirichletBC(self.get_iu(), self.dr_expr, RVE_boundaries, method='pointwise')
        self.revise_BCs(remove=True, new_BCs=[hom_bc], _as='hom') # NOTE: here the _as='hom' is another thing than the homogenization :) !
        
    def update_case(self, case):
        HomogenizationElastic.update_case(self, case)
        
        if self.dep_dim == 1:
            raise NotImplementedError()
            
        elif self.dep_dim == 2:
            assert case < 3
            self._Eps_Voigt[:] = 0.
            self._Eps_Voigt[case] = self.strain_scalar
            
            self._Eps_Voigt[2] *= 0.5
            
        elif self.dep_dim == 3:
            assert case < 6
            self._Eps_Voigt[:] = 0.
            self._Eps_Voigt[case] = self.strain_scalar
            
            for i in [3, 4, 5]:
                self._Eps_Voigt[i] *= 0.5
        
        self.m_strain.assign(df.Constant(self._Eps_Voigt))
    
    def get_sigma(self):
        return self.mat.sigma(self.get_uu())