from feniQS.structure.structures_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_loadings import *
from feniQS.structure.helper_BCs import ds_at_end_point_of_interval_mesh, pth_helper_BCs

pth_struct_peerling1996 = CollectPaths('./feniQS/structure/struct_peerling1996.py')
pth_struct_peerling1996.add_script(pth_structures_fenics)
pth_struct_peerling1996.add_script(pth_fenics_functions)
pth_struct_peerling1996.add_script(pth_helper_loadings)
pth_struct_peerling1996.add_script(pth_helper_BCs)

class ParsPeerling(ParsBase):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Default values are set
            ## GEOMETRY
            self.geo_scale = 1.
            self.L = 100 * self.geo_scale
            
            ## Imperfection due to smaller cross section
            self.W = 10 * self.geo_scale # length of the bar with reduced E
            self.x_from = (self.L - self.W) / 2
            self.x_to = (self.L + self.W) / 2
            self.red_factor = 0.9
            
            ## MESH
            self._res = 50
            
            ## LOADs and BCs
            self.loading_control = 'u'
            self.loading_level = 0.2 * self.geo_scale
            self.loading_case = 'ramp'
            self.loading_scales = [1]
            self.loading_N = 0.4 # only relevant if case='sin' or 'zigzag'
            self.loading_T = 1.0; self.loading_t_end=1.0
            
            ## REST
            self._write_files = True
            self._plot = True
        
        else: # Get from a dictionary
            self.__dict__.update(kwargs)

class Peerling1996(StructureFEniCS):
    def __init__(self, pars, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsPeerling(**pars)
        if _name is None:
            _name = 'peerling1996'
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_structure(self):
        ### MESH ###
        self.mesh = df.IntervalMesh(self.pars._res, 0., self.pars.L)
        
        if self.pars._plot:
            plt.figure()
            df.plot(self.mesh)
            plt.title(f"Mesh of '{self._name}'")
            plt.savefig(self._path + 'meshView.png', bbox_inches='tight', dpi=300)
            plt.show()
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
        
        ### LOADs ###
        self.u_right = None # by default
        self.f_right = None # by default
        l = time_varying_loading(fs=self.pars.loading_level*np.array(self.pars.loading_scales), N=self.pars.loading_N \
                                       , T=self.pars.loading_T, _case=self.pars.loading_case \
                                           , _plot=self.pars._plot, lab_y=self.pars.loading_control.lower(), _path=self._path)
        if self.pars.loading_control.lower() == 'u':
            self.u_right = l
        elif self.pars.loading_control.lower() == 'f':
            self.f_right = l
        else:
            raise ValueError(f"Loading control of the structure is not recognized.")
    
    def get_BCs(self, i_u):
        tol = self.mesh.rmin() / 1000
        def left(x):
            return df.near(x[0], 0., tol)
        bcl, bcl_dofs = boundary_condition(i_u, df.Constant(0.0), left)
        bcs_DR = {'left': {'bc': bcl, 'bc_dofs': bcl_dofs}}
        if self.u_right is not None:
            ll = self.u_right
            def right(x):
                return df.near(x[0], self.pars.L, tol)
            bcr, bcr_dofs = boundary_condition(i_u, self.u_right, right)
            bcs_DR_inhom = {'right': {'bc': bcr, 'bc_dofs': bcr_dofs}}
        elif self.f_right is not None:
            ll = self.f_right
            bcs_DR_inhom = {}
        time_varying_loadings = {'right': ll}
        
        return bcs_DR, bcs_DR_inhom, time_varying_loadings
    
    def get_reaction_nodes(self, reaction_places):
        nodes = []
        tol = self.mesh.rmin() / 1000
        for rp in reaction_places:
            if 'left' in rp.lower():
                ps = [[0.]] # left
            elif 'right' in rp.lower():
                ps = [[self.pars.L]] # right
            else:
                raise NameError('Reaction place is not recognized.')
            nodes.append(ps)
        return nodes # A list of lists each being nodes related to a reaction place
    
    def get_tractions_and_dolfin_measures(self):
        ts = [] # Tractions
        dss = [] # Measures
        if self.f_right is not None:
            ts.append(df.as_tensor((self.f_right)))
            dss.append(ds_at_end_point_of_interval_mesh(self.mesh, self.pars.L))
        return ts, dss
    
    def get_tractions_dofs(self, i_full, i_u):
        dofs = []
        if self.f_right is not None:
            nodes = [[self.pars.L]] # at right side
            dofs = dofs_at(points=nodes, V=i_full, i=i_u)
        return dofs
    
    def build_special_fenics_fields(self):
        self.special_fenics_fields = {}
        if self.pars.red_factor==1 or self.pars.x_from==self.pars.x_to:
            self.special_fenics_fields['sigma_scale'] = 1.0
            print("WARNING: No geometric imperfection is considered in the bar (peerling1996 structure).")
        else:
            self.special_fenics_fields['sigma_scale'] = df.Expression('(x[0] >= x_from && x[0] <= x_to) ? sc : 1.0', \
                             x_from=self.pars.x_from, x_to=self.pars.x_to, \
                                 sc=self.pars.red_factor, degree=0)

    