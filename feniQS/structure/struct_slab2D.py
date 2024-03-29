from feniQS.structure.structures_fenics import *
from feniQS.structure.helper_mesh_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_loadings import *
from feniQS.structure.helper_BCs import *

pth_struct_slab2d = CollectPaths('./feniQS/structure/struct_slab2D.py')
pth_struct_slab2d.add_script(pth_structures_fenics)
pth_struct_slab2d.add_script(pth_helper_mesh_fenics)
pth_struct_slab2d.add_script(pth_fenics_functions)
pth_struct_slab2d.add_script(pth_helper_loadings)
pth_struct_slab2d.add_script(pth_helper_BCs)

class ParsSlab2D(ParsBase):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            ## GEOMETRY
            self.lx = 1.
            self.ly = 1.
            
            ## MESH
            self.res_x = 1
            self.res_y = 1
            self.el_size_max = None
            self.el_size_min = None
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

            self.bc_y_at_left = 'corner' # or 'edge'

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
    
    @staticmethod
    def get_regular_grid_points(pars, finement=2.0, x_range=None, include_edges=True):
        if x_range is None:
            _distance = pars.lx - 0.
            x_range = [0. + 0.1 * _distance, pars.lx - 0.1 * _distance] # 80% of total length
        x1, x2 = x_range
        _res_y = int(pars.res_y * finement)
        _res_x = int(_res_y * (x2-x1) / pars.ly)
        xs = np.linspace(x1, x2, _res_x)
        _dy = 0.5 * pars.ly / _res_y
        ys = np.linspace(_dy, pars.ly - _dy, _res_y)
        if include_edges:
            ys = np.array([0.] + list(ys) + [pars.ly])
        xv, yv = np.meshgrid(xs, ys)
        _points = np.vstack([xv.ravel(), yv.ravel()]).T
        return _points

class Slab2D(StructureFEniCS):
    def __init__(self, pars, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsBase(**pars)
        if _name is None:
            _name = 'slab2d'
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_mesh(self):
        self.embedded_nodes = np.array(self.pars.embedded_nodes)
        mesh = slab2D_mesh(lx=self.pars.lx, ly=self.pars.ly \
                           , res_x=self.pars.res_x, res_y=self.pars.res_y
                           , embedded_nodes=self.embedded_nodes \
                           , el_size_min=self.pars.el_size_min, el_size_max=self.pars.el_size_max \
                           , _path=self._path)
        ParsSlab2D.set_embedded_nodes(self.pars, self.embedded_nodes) # since self.embedded_nodes might have been adjusted (veryyyyy slightly) in mesh generation process.
        return mesh

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
    
    def get_time_varying_loadings(self):
        time_varying_loadings = {}
        if self.u_x is not None:
            time_varying_loadings['x_right'] = self.u_x
        elif self.f_x is not None:
            time_varying_loadings['x_right'] = self.f_x
        return time_varying_loadings

    def build_BCs(self, i_u):
        assert i_u.num_sub_spaces() == 2
        tol = self.mesh.rmin() / 1000.
        def _left(x, on_boundary):
            return df.near(x[0], 0., tol)
        
        self.bcs_DR, self.bcs_DR_inhom = {}, {}
        
        bc_left_x, bc_left_x_dofs = boundary_condition(i_u.sub(0), df.Constant(0.0), _left)
        self.bcs_DR.update({'left_x': {'bc': bc_left_x, 'bc_dofs': bc_left_x_dofs}})

        if 'corner' in self.pars.bc_y_at_left.lower():
            # CASE-1: fixed in corner
            def _left_bot(x, on_boundary):
                return df.near(x[0], 0., tol) and df.near(x[1], 0., tol)
            bc_left_y, bc_left_y_dofs = boundary_condition_pointwise(i_u.sub(1), df.Constant(0.0), _left_bot)
        elif 'edge' in self.pars.bc_y_at_left.lower():
            # CASE-2: fully clamped
            bc_left_y, bc_left_y_dofs = boundary_condition(i_u.sub(1), df.Constant(0.0), _left)
        else:
            raise ValueError(f"The parameter 'pars.bc_y_at_left={self.pars.bc_y_at_left}' is not recognized. Set it to either 'corner' or 'edge'.")
        self.bcs_DR.update({'left_y': {'bc': bc_left_y, 'bc_dofs': bc_left_y_dofs}})
        
        if self.u_x is not None:
            def _right(x, on_boundary):
                return df.near(x[0], self.pars.lx, tol)
            bc_right_x, bc_right_x_dofs = boundary_condition(i_u.sub(0), self.u_x, _right)
            self.bcs_DR_inhom.update({'right_x': {'bc': bc_right_x, 'bc_dofs': bc_right_x_dofs}})
    
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
        if self.f_x is not None:
            f_0 = df.Constant(0.0)
            ts.append(df.as_tensor((self.f_x, f_0)))
            dss.append(ds_on_rectangle_mesh(self.mesh, x_from=self.pars.lx, x_to=self.pars.lx \
                                            , y_from=0., y_to=self.pars.ly))
        return ts, dss
    
    def get_tractions_dofs(self, i_full, i_u):
        dofs = []
        if self.f_x is not None:
            nodes = self.get_reaction_nodes(['right'])[0]
            dofs = dofs_at(points=nodes, V=i_full, i=i_u.sub(0)) # in x-direction
        return dofs

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
    
    def get_regular_grid_points(self, finement=2.0, x_range=None, include_edges=True):
        return ParsSlab2D.get_regular_grid_points(pars=self.pars, finement=finement, x_range=x_range, include_edges=include_edges)

    def get_window_of_nodes(self, x_range=None, y_range=None, sparse=1):
        """
        Returns every sparse-th node from a subset of mesh nodes located within a rectangle surrounded by x_range and y_range.
        The x_range and y_range by default covers 4 surrounding edges of the beam.
        """
        nodes = self.mesh.coordinates() # all nodes
        ids_ = []
        if x_range is None:
            x_range = [0., self.pars.lx]
        if y_range is None:
            y_range = [0., self.pars.ly]
        for ic, c in enumerate(nodes):
            if (x_range[0]<=c[0]<=x_range[1]) and (y_range[0]<=c[1]<=y_range[1]):
                ids_.append(ic)
        ids_ = ids_[::sparse]
        return nodes[ids_, :]

    def plot_mesh(self, ax=None, _save=True, _path=None):
        rm = self.mesh.rmin(); dpi = min(500, 200. / rm)
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