from feniQS.structure.structures_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_mesh_fenics import *
from feniQS.structure.helper_loadings import *

pth_struct_rectangle2d = CollectPaths('./feniQS/structure/struct_rectangle2D.py')
pth_struct_rectangle2d.add_script(pth_structures_fenics)
pth_struct_rectangle2d.add_script(pth_fenics_functions)
pth_struct_rectangle2d.add_script(pth_helper_loadings)

class ParsLshape2D(ParsBase):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            ## GEOMETRY
            _scale = 1.
            self.lx = _scale*10. # total width (in x direction)
            self.ly = _scale*10. # total height (in y direction)
            self.wx = _scale*5. # non-empty width (in x direction)
            self.wy = _scale*5. # non-empty height (in y direction)
            
            ## MESH
            self.resolutions = {'res_x': 10, # total resolution in x direction
                                'res_y': 10, # total resolution in y direction
                                'embedded_nodes': (),
                                'refinement_level': 0,
                                'el_size_max': None}
            
            ## LOADs and BCs
            self.loading_control = 'u_pull' # or 'u_shear' or 'f_pull' or 'f_shear'
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
        pars.resolutions['embedded_nodes'] = tuple(tuple(float(c) for c in p) for p in embedded_nodes)
            # Can this be done via setattr and getattr methods (as we here change value of a dictionary) ?
    
    @staticmethod
    def is_inside(pars, p, include_edges=True):
        x, y = p[0], p[1]
        if include_edges:
            _is_out = (x>pars.wx) and (y<(pars.ly-pars.wy))
        else:
            _is_out = (x>=pars.wx) and (y<=(pars.ly-pars.wy))
            _is_out = _is_out or (x==0.) or (y==0.) or (x==pars.lx) or (y==pars.ly)
        return (not _is_out)

    @staticmethod
    def get_regular_grid_points(pars, finement=1.0, portion=0.9):
        _res_x = int(pars.resolutions['res_x'] * finement)
        _res_y = int(pars.resolutions['res_y'] * finement)
        x1 = 0.5 * (1. - portion) * pars.lx
        x2 = pars.lx - x1
        xs = np.linspace(x1, x2, _res_x)
        y1 = 0.5 * (1. - portion) * pars.ly
        y2 = pars.ly - y1
        ys = np.linspace(y1, y2, _res_y)
        xv, yv = np.meshgrid(xs, ys)
        _points = np.vstack([xv.ravel(), yv.ravel()]).T
        ## Exclude possible points locating in the notch area
        rows_to_delete = []
        for i in range(_points.shape[0]):
            if not ParsLshape2D.is_inside(pars, _points[i,:], include_edges=False):
                rows_to_delete.append(i)
        points_regular = np.delete(_points, rows_to_delete, 0)
        return points_regular

class Lshape2D(StructureFEniCS):
    def __init__(self, pars, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsLshape2D(**pars)
        if _name is None:
            _name = 'Lshape2d'
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_mesh(self):
        self.embedded_nodes = np.array(self.pars.resolutions['embedded_nodes'])
        mesh = L_shape_mesh(lx=self.pars.lx, ly=self.pars.ly \
                            , wx=self.pars.wx, wy=self.pars.wy \
                            , res_x=self.pars.resolutions['res_x'], res_y=self.pars.resolutions['res_y'] \
                            , embedded_nodes=self.embedded_nodes, el_size_max=self.pars.resolutions['el_size_max'] \
                            , _path=self._path)
        ParsLshape2D.set_embedded_nodes(self.pars, self.embedded_nodes) # since self.embedded_nodes might have been adjusted (veryyyyy slightly) in mesh generation process.
        return mesh

    def _build_structure(self, _build_load=True):
        ### MESH ###
        self.mesh = self._build_mesh()
        self.refine_mesh(self.pars.resolutions['refinement_level'])
        if self.pars._plot:
            self.plot_mesh(_save=True)
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
        
        self.u_x_side = None # by default
        self.f_x_side = None # by default
        self.u_y_side = None # by default
        self.f_y_side = None # by default
        ### LOADs ###
        if _build_load:
            l = time_varying_loading(fs=self.pars.loading_level*np.array(self.pars.loading_scales) \
                                                      , N=self.pars.loading_N, T=self.pars.loading_T, _case=self.pars.loading_case \
                                                      , _plot=self.pars._plot, lab_y=self.pars.loading_control.lower(), _path=self._path)
            lc = self.pars.loading_control.lower()
            if lc=='u_pull':
                self.u_x_side = l
            elif lc=='u_shear':
                self.u_y_side = l
            elif lc=='f_pull':
                self.f_x_side = l
            elif lc=='f_shear':
                self.f_y_side = l
            else:
                raise ValueError(f"Loading control of the structure is not recognized.")
    
    def get_time_varying_loadings(self):
        time_varying_loadings = {}
        if self.u_x_side is not None:
            time_varying_loadings['x_side'] = self.u_x_side
        elif self.f_x_side is not None:
            time_varying_loadings['x_side'] = self.f_x_side
        elif self.u_y_side is not None:
            time_varying_loadings['y_side'] = self.u_y_side
        elif self.f_y_side is not None:
            time_varying_loadings['y_side'] = self.f_y_side
        return time_varying_loadings

    def build_BCs(self, i_u):
        assert i_u.num_sub_spaces() == 2
        tol = self.mesh.rmin() / 1000.
        self.bcs_DR, self.bcs_DR_inhom = {}, {}
        
        ## Clamp the bottom edge entirely
        def bot_edge(x, on_boundary):
            return on_boundary and df.near(x[1], 0., tol)
        bc_be_x, bc_be_x_dofs = boundary_condition(i_u.sub(0), df.Constant(0.0), bot_edge)
        bc_be_y, bc_be_y_dofs = boundary_condition(i_u.sub(1), df.Constant(0.0), bot_edge)
        self.bcs_DR.update({'bot_x': {'bc': bc_be_x, 'bc_dofs': bc_be_x_dofs}})
        self.bcs_DR.update({'bot_y': {'bc': bc_be_y, 'bc_dofs': bc_be_y_dofs}})
        
        _u_side = None
        if self.u_x_side is not None:
            _u_side = self.u_x_side
            i_side = i_u.sub(0)
            l_side = 'side_x'
        elif self.u_y_side is not None:
            _u_side = self.u_y_side
            i_side = i_u.sub(1)
            l_side = 'side_y'
        if _u_side is not None:
            def side_edge(x, on_boundary):
                return on_boundary and df.near(x[0], self.pars.lx, tol)
            bc_side, bc_side_dofs = boundary_condition(i_side, _u_side, side_edge)
            self.bcs_DR_inhom.update({l_side: {'bc': bc_side, 'bc_dofs': bc_side_dofs}})
    
    def get_reaction_nodes(self, reaction_places):
        nodes = []
        tol = self.mesh.rmin() / 1000.
        cs = self.mesh.coordinates()
        for rp in reaction_places:
            if 'bot' in rp.lower():
                ps = cs[abs(cs[:,1])<tol]
            elif 'side' in rp.lower():
                ps = cs[abs(cs[:,0]-self.pars.lx)<tol]
            else:
                raise NameError('Reaction place is not recognized. Specify the side (left or right).')
            nodes.append(ps)
        return nodes # A list of lists each being nodes related to a reaction place
    
    def get_support_nodes(self):
        tol = self.mesh.rmin() / 1000.
        cs = self.mesh.coordinates()
        return cs[abs(cs[:,1])<tol]
    
    def get_tractions_and_dolfin_measures(self):
        ts = [] # Tractions
        dss = [] # Measures
        _tx = df.Constant(0.0); _ty = df.Constant(0.0); bb = False
        if self.f_x_side is not None:
            _tx = self.f_x_side; bb=True
        if self.f_y_side is not None:
            _ty = self.f_y_side; bb=True
        if bb:
            ts = [df.as_tensor((_tx, _ty))]
            tol = self.mesh.rmin() / 1000.
            def side_edge(x, on_boundary):
                    return on_boundary and df.near(x[0], self.pars.lx, tol)
            dom = df.AutoSubDomain(side_edge)
            mf = df.MeshFunction('size_t', self.mesh, 1) # 1 goes to edges
            mf.set_all(0)
            dom.mark(mf, 1)
            ds = df.Measure('ds', domain=self.mesh, subdomain_data=mf)
            dss = [ds(1)]
        return ts, dss
    
    def switch_loading_control(self, loading_control, load=None):
        """
        loading_control (at the side):
            'u_pull' : Displacement-controlled in x direction (perpendicular to the side)
            'f_pull' : Force-controlled in x direction (perpendicular to the side)
            Replacing 'pull' with 'shear' makes the same controls but in y direction (parallel to the side).
        """
        _possible_loads = (self.u_x_side, self.u_y_side, self.f_x_side, self.f_y_side)
        if load is None:
            load = next(v for v in _possible_loads if v is not None)
            if load is None:
                raise ValueError(f"No loading is given or defined before for the structure.")
        if loading_control.lower() == 'u_pull':
            self.u_x_side = load
            self.u_y_side = self.f_x_side = self.f_y_side = None
        elif loading_control.lower() == 'u_shear':
            self.u_y_side = load
            self.u_x_side = self.f_x_side = self.f_y_side = None
        elif loading_control.lower() == 'f_pull':
            self.f_x_side = load
            self.u_x_side = self.u_y_side = self.f_y_side = None
        elif loading_control.lower() == 'f_shear':
            self.f_y_side = load
            self.u_x_side = self.u_y_side = self.f_x_side = None
        else:
            raise ValueError(f"Loading control is not recognized. Possible values are '{_possible_loads}' .")
        self.pars.loading_control = loading_control

    def plot_mesh(self, ax=None, _save=True, _path=None):
        rm = self.mesh.rmin(); dpi = min(500, 200. / rm)
        if ax is None:
            fig, ax = plt.subplots()
        df.plot(self.mesh, color='gray', linewidth=0.2)
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
    pars = ParsLshape2D()
    # pars.wx=0.3*pars.lx
    # pars.wy=0.6*pars.ly
    pars.resolutions['res_x'] = pars.resolutions['res_y'] = 28
    aa = ParsLshape2D.get_regular_grid_points(pars, finement=0.3, portion=0.9)
    ParsLshape2D.set_embedded_nodes(pars, aa)
    struct = Lshape2D(pars)