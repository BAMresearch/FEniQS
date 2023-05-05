from feniQS.structure.structures_fenics import *
from feniQS.structure.helper_mesh_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_loadings import *

pth_struct_slab2d = CollectPaths('struct_slab2D.py')
pth_struct_slab2d.add_script(pth_structures_fenics)
pth_struct_slab2d.add_script(pth_helper_mesh_fenics)
pth_struct_slab2d.add_script(pth_fenics_functions)
pth_struct_slab2d.add_script(pth_helper_loadings)

class ParsSlab2D(ParsBase):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            ## GEOMETRY
            self.lx = 1.
            self.ly = 1.
            
            ## MESH
            self.res_x = 1
            self.res_y = 1
            embedded_nodes = [[0.25, 0.5], [0.5, 0.25], [0.75, 0.5], [0.5, 0.75]]
            ParsSlab2D.set_embedded_nodes(self, embedded_nodes)
            
            ## LOADs and BCs
            self.loading_control = 'u'
            self.loading_level = self.lx / 100
            
            self.loading_scales = [1]
            self.loading_case = 'ramp'
            self.loading_N = 1.0 # only relevant if case=sin or case=zigzag
            self.loading_T = 1.0
            self.loading_t_end = self.loading_T

            ## REST
            self._write_files = True
            self._plot = True
            
        else: # Get from a dictionary
            ParsBase.__init__(self, **kwargs)
    
    @staticmethod
    def set_embedded_nodes(pars, embedded_nodes):
        """
        Given embedded_nodes must be converted to tuple of tuples.
            (to be hashable and able to be writtin/read to/from yaml file)
        """
        pars.embedded_nodes = tuple(tuple(float(c) for c in p) for p in embedded_nodes)

class Slab2D(StructureFEniCS):
    def __init__(self, pars, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsBase(**pars)
        if _name is None:
            _name = 'slab2d'
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_mesh(self):
        self.embedded_nodes = np.array(self.pars.embedded_nodes)
        return slab2D_mesh(lx=self.pars.lx, ly=self.pars.ly \
                           , res_x=self.pars.res_x, res_y=self.pars.res_y
                           , embedded_nodes=self.embedded_nodes, _path=self._path)
        ParsBend3Point2D.set_embedded_nodes(self.pars, self.embedded_nodes) # since self.embedded_nodes might have been adjusted (veryyyyy slightly) in mesh generation process.

    def _build_structure(self, _build_load=True):
        ### MESH ###
        self.mesh = self._build_mesh()
        if self.pars._plot:
            self.plot_mesh(_save=True)
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
        
        self.u_x = None # by default
        self.f_x = None # by default
        ### LOADs ###
        if _build_load:
            l = time_varying_loading(fs=self.pars.loading_level*np.array(self.pars.loading_scales) \
                                                      , N=self.pars.loading_N, T=self.pars.loading_T, _case=self.pars.loading_case \
                                                      , _plot=self.pars._plot, lab_y=self.pars.loading_control.lower(), _path=self._path)
            if self.pars.loading_control.lower() == 'u':
                self.u_x = l
            elif self.pars.loading_control.lower() == 'f':
                self.f_x = l
            else:
                raise ValueError(f"Loading control of the structure is not recognized.")
    
    def get_BCs(self, i_u):
        assert i_u.num_sub_spaces() == 2
        time_varying_loadings = {}
        if self.u_x is not None:
            time_varying_loadings.update({'x_right': self.u_x})
        if self.f_x is not None:
            time_varying_loadings.update({'x_right': self.f_x})
        
        tol = self.mesh.rmin() / 1000.
        def _left(x, on_boundary):
            return df.near(x[0], 0., tol)
        
        bcs_DR, bcs_DR_inhom = {}, {}
        
        bc_left, bc_left_dofs = boundary_condition(i_u, df.Constant((0.0, 0.0)), _left)
        bcs_DR.update({'left': {'bc': bc_left, 'bc_dofs': bc_left_dofs}})
        
        if self.u_x is not None:
            def _right(x, on_boundary):
                return df.near(x[0], self.pars.lx, tol)
            bc_right_x, bc_right_x_dofs = boundary_condition(i_u.sub(0), self.u_x, _right)
            bcs_DR_inhom.update({'right_x': {'bc': bc_right_x, 'bc_dofs': bc_right_x_dofs}})
        
        return bcs_DR, bcs_DR_inhom, time_varying_loadings
    
    def get_reaction_nodes(self, reaction_places):
        nodes = []
        tol = self.mesh.rmin() / 1000.
        for rp in reaction_places:
            if 'left' in rp.lower():
                xx = 0.
            elif 'right' in rp.lower():
                xx = self.pars.lx
            else:
                raise NameError('Reaction place is not recognized. Specify the side (left or right).')
            ps = []
            for c in self.mesh.coordinates():
                if abs(c[0]-xx) < tol:
                    ps.append(c)
            nodes.append(ps)
        return nodes # A list of lists each being nodes related to a reaction place
    
    def get_tractions_and_dolfin_measures(self):
        ts = [] # Tractions
        dss = [] # Measures
        return ts, dss
    
    def switch_loading_control(self, loading_control, load=None):
        """
        loading_control:
            'u' : Displacement-controlled
            'f' : Force-controlled
        """
        if load is None:
            if self.u_x is None:
                load = self.f_x
            else:
                load = self.u_x
            if load is None:
                raise ValueError(f"No loading is given or defined before for the structure.")
        if loading_control.lower() == 'u':
            self.u_x = load
            self.f_x = None
        elif loading_control.lower() == 'f':
            self.f_x = load
            self.u_x = None
        else:
            raise ValueError(f"Loading control is not recognized. Possible values are 'u' and 'f' .")
        self.pars.loading_control = loading_control
    
    def plot_mesh(self, ax=None, _save=True, _path=None):
        rm = self.mesh.rmin(); dpi = max(500, 200. / rm)
        if ax is None:
            fig, ax = plt.subplots()
        df.plot(self.mesh, color='gray', linewidth=min(1., 5*rm))
        if len(self.embedded_nodes) > 0:
            ax.plot(self.embedded_nodes[:,0], self.embedded_nodes[:,1], marker='.', fillstyle='none' \
                     , linestyle='', color='red', label='Embedded nodes')
            ax.legend()
        ax.set_title(f"Mesh of '{self._name}'")
        if _save:
            if _path is None:
                _path = self._path
            plt.savefig(_path + 'meshView.png', bbox_inches='tight', dpi=dpi)
            plt.show()
        return ax, {'dpi': dpi}

if __name__=='__main__':
    pars = ParsSlab2D()
    struct = Slab2D(pars)