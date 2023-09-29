from feniQS.structure.structures_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_mesh_fenics import *
from feniQS.structure.helper_BCs import *
from feniQS.structure.helper_loadings import *

pth_struct_bend3point2d = CollectPaths('./feniQS/structure/struct_bend3point2d.py')
pth_struct_bend3point2d.add_script(pth_structures_fenics)
pth_struct_bend3point2d.add_script(pth_fenics_functions)
pth_struct_bend3point2d.add_script(pth_helper_mesh_fenics)
pth_struct_bend3point2d.add_script(pth_helper_BCs)
pth_struct_bend3point2d.add_script(pth_helper_loadings)

class ParsBend3Point2D(ParsBase):
    def __init__(self, **kwargs):
        """
        It must have the following parameters:
            ## GEOMETRY
            self.lx
            self.ly
            self.l_notch # Width of notch
            self.h_notch # Hight of notch
            self.left_notch # Left edge of notch (the x coordinate)
            self.left_sup # The x coordinate (center)
            self.right_sup # The x coordinate (center)
            self.left_sup_w # The width of left-support
            self.right_sup_w # The width of right support
            self.x_from # Starting 'x' of oading at the middle
            self.x_to # Ending 'x' of loading at the middle
            self.cmod_left # Left sensor position of CMOD (x coordinate)
            self.cmod_right # Right sensor position of CMOD (x coordinate)
            
            ## MESH
            self.resolutions = {'res_y': int, 'scale': float, 'embedded_nodes': tuple, 'refinement_level': int, el_size_max: float}
                # 'res_y': the number of elements at vertical edges of the beam (at each of left/right sides).
                    # This together with 'scale' will determine the resolutions at remaining edges.
                # 'scale': a factor by which the element sizes are scaled towards the center of the beam.
                    --> A 'scale' smaller than one implies finer mesh sizes towards the center.
                    # Note: The 'scale' is total and symmetric, i.e.:
                        # element size at the very center = 'scale' * element size at the very sides.
                # 'embedded_nodes': A set of node-coordinates which we embed in the mesh.
                    By default is empty (). It must be 'tuple' so that it can be written to yaml file and is also hashable.
                # 'refinement_level': A non-negative integer.
                    By default is 0, meaning no refinement.
                    For refinement_level>0, the original mesh is subjected to df.refine() routine for 'refinement_level' times.
                # 'el_size_max' : A float dpecifying the maximum element size (when a certain level of refined mesh is required).
            
            ## LOADs and BCs
            self.fix_x_at # 'left' or 'right'
            self.loading_control # Either 'u': displacement-controlled or 'f': force-controlled
            self.loading_level # A total level (magnifier) of loading (at middle), which can be either displacement or force (depending on self.loading_control)
            self.loading_scales # Several possible peaks for loading (at middle) --> peaks = level * scales
            self.loading_case # 'ramp' or 'sin' or 'zigzag'
            self.loading_N # (=1) Number of periods for case='sin' or case='zigzag'
            self.loading_T # (=1.0) Maximum 't' of DEFINED loading
            self.loading_t_end # (=self.loading_T) Maximum 't' of APPLIED loading
            
            ## REST
            self._write_files = True
            self._plot = True
            
            ## Possible other parameters (e.g. for a structure based on DIC data)
        """
        if len(kwargs)==0: # Some values are set
            raise NotImplementedError("Overwrite the constructor of the base class 'ParsBend3Point2D'.")
        else:
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
            return x>=0.0 and x<=pars.lx and y>=0.0 and y<=pars.ly \
                and not(x>(pars.lx-pars.l_notch)/2 and x<(pars.lx+pars.l_notch)/2 and y>=0.0 and y<pars.h_notch)
        else:
            return x>0.0 and x<pars.lx and y>0.0 and y<pars.ly \
                and not(x>=(pars.lx-pars.l_notch)/2 and x<=(pars.lx+pars.l_notch)/2 and y>=0.0 and y<=pars.h_notch)
    
    @staticmethod
    def get_regular_grid_points(pars, finement=2.0, x_range=None, include_edges=True):
        if x_range is None:
            supports_distance = pars.right_sup - pars.left_sup
            x_range = [pars.left_sup + 0.1 * supports_distance, pars.right_sup - 0.1 * supports_distance] # 80% of support distance
        x1, x2 = x_range
        _res_y = int(pars.resolutions['res_y'] * finement)
        _res_x = int(_res_y * (x2-x1) / pars.ly)
        xs = np.linspace(x1, x2, _res_x)
        _dy = 0.5 * pars.ly / _res_y
        ys = np.linspace(_dy, pars.ly - _dy, _res_y)
        if include_edges:
            ys = np.array([0] + list(ys) + [pars.ly])
        xv, yv = np.meshgrid(xs, ys)
        _points = np.vstack([xv.ravel(), yv.ravel()]).T
        ## Exclude possible points locating in the notch area
        rows_to_delete = []
        for i in range(_points.shape[0]):
            p = _points[i,:]
            if not ParsBend3Point2D.is_inside(pars, p):
                rows_to_delete.append(i)
        points_regular = np.delete(_points, rows_to_delete, 0)
        return points_regular

class Bend3Point2D(StructureFEniCS):
    def __init__(self, pars, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsBend3Point2D(**pars)
        if _name is None:
            _name = 'bend3point2d'
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_mesh(self):
        self.embedded_nodes = np.array(self.pars.resolutions['embedded_nodes'])
        mesh = notched_rectangle_mesh(lx=self.pars.lx, ly=self.pars.ly \
                                      , load_Xrange=[self.pars.x_from, self.pars.x_to] \
                                      , l_notch=self.pars.l_notch, h_notch=self.pars.h_notch \
                                      , c_notch=self.pars.left_notch+self.pars.l_notch/2 \
                                      , left_sup=self.pars.left_sup, right_sup=self.pars.right_sup \
                                      , left_sup_w=self.pars.left_sup_w, right_sup_w=self.pars.right_sup_w \
                                      , res_y=self.pars.resolutions['res_y'], scale=self.pars.resolutions['scale'] \
                                      , embedded_nodes=self.embedded_nodes, el_size_max=self.pars.resolutions['el_size_max'] \
                                      , _path=self._path)
        ParsBend3Point2D.set_embedded_nodes(self.pars, self.embedded_nodes) # since self.embedded_nodes might have been adjusted (veryyyyy slightly) in mesh generation process.
        return mesh
    
    def _build_structure(self, _build_middle_load=True):
        ### MESH ###
        self.mesh = self._build_mesh()
        self.refine_mesh(self.pars.resolutions['refinement_level'])
        if self.pars._plot:
            self.plot_mesh(_save=True)
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
        
        self.u_y_middle = None # by default
        self.f_y_middle = None # by default
        ### LOADs ###
        if _build_middle_load:
            l = time_varying_loading(fs=self.pars.loading_level*np.array(self.pars.loading_scales) \
                                                      , N=self.pars.loading_N, T=self.pars.loading_T, _case=self.pars.loading_case \
                                                      , _plot=self.pars._plot, lab_y=self.pars.loading_control.lower(), _path=self._path)
            if self.pars.loading_control.lower() == 'u':
                self.u_y_middle = l
            elif self.pars.loading_control.lower() == 'f':
                self.f_y_middle = l
            else:
                raise ValueError(f"Loading control of the structure is not recognized.")
    
    def get_BCs(self, i_u):
        time_varying_loadings = {}
        if self.u_y_middle is not None:
            time_varying_loadings.update({'y_middle': self.u_y_middle})
        bcs_DR, bcs_DR_inhom = load_and_bcs_on_3point_bending(self.mesh, self.pars.lx, self.pars.ly, self.pars.x_from, self.pars.x_to \
                                          , i_u=i_u, u_expr=self.u_y_middle \
                                              , left_sup=self.pars.left_sup, right_sup=self.pars.right_sup \
                                                  , left_sup_w=self.pars.left_sup_w, right_sup_w=self.pars.right_sup_w, x_fix=self.pars.fix_x_at)
        if self.f_y_middle is not None:
            time_varying_loadings.update({'y_middle': self.f_y_middle})
        return bcs_DR, bcs_DR_inhom, time_varying_loadings
    
    def get_reaction_nodes(self, reaction_places):
        nodes = []
        tol = self.mesh.rmin() / 1000
        for rp in reaction_places:
            if 'left' in rp.lower():
                ps = [list(p) for p in self.get_support_nodes('left')] # left
            elif 'right' in rp.lower():
                ps = [list(p) for p in self.get_support_nodes('right')] # right
            elif 'middle' in rp.lower():
                ps = []
                for pp in self.mesh.coordinates():
                    if pp[0]>=self.pars.x_from-tol and pp[0]<=self.pars.x_to+tol and abs(pp[1]-self.pars.ly)<tol:
                        ps.append(pp)
            else:
                raise NameError('Reaction place is not recognized.')
            nodes.append(ps)
        return nodes # A list of lists each being nodes related to a reaction place
    
    def get_support_nodes(self, which='both'):
        """
        which is either of:
            'left'
            'right'
            'both' (default)
        """
        tol = self.mesh.rmin() / 1000.
        cs = self.mesh.coordinates()
        ps_bot = cs[abs(cs[:,1])<tol]
        ps_left = np.empty((0, 2)); ps_right = ps_left
        if any(w in which.lower() for w in ['both', 'left']): # left
            b1 = self.pars.left_sup-self.pars.left_sup_w/2. - tol <= ps_bot[:,0]
            b2 = ps_bot[:,0] <= self.pars.left_sup+self.pars.left_sup_w/2. + tol
            b = [bb1 and bb2 for bb1,bb2 in zip(b1,b2)]
            ps_left = ps_bot[b]
        if any(w in which.lower() for w in ['both', 'right']): # right
            b1 = self.pars.right_sup-self.pars.right_sup_w/2. - tol <= ps_bot[:,0]
            b2 = ps_bot[:,0] <= self.pars.right_sup+self.pars.right_sup_w/2. + tol
            b = [bb1 and bb2 for bb1,bb2 in zip(b1,b2)]
            ps_right = ps_bot[b]
        return np.concatenate((ps_left, ps_right), axis=0)
    
    def get_tractions_and_dolfin_measures(self):
        ts = [] # Tractions
        dss = [] # Measures
        if self.f_y_middle is not None:
            f_0 = df.Constant(0.0)
            ts.append(df.as_tensor((f_0, self.f_y_middle)))
            dss.append(ds_on_rectangle_mesh(self.mesh, x_from=self.pars.x_from, x_to=self.pars.x_to \
                                            , y_from=self.pars.ly, y_to=self.pars.ly))
        return ts, dss
    
    def get_tractions_dofs(self, i_full, i_u):
        dofs = []
        if self.f_y_middle is not None:
            nodes = self.get_reaction_nodes(['middle'])[0]
            dofs = dofs_at(points=nodes, V=i_full, i=i_u.sub(1)) # in y-direction
        return dofs
    
    def switch_loading_control(self, loading_control, load=None):
        """
        loading_control:
            'u' : Displacement-controlled
            'f' : Force-controlled
        """
        if load is None:
            if self.u_y_middle is None:
                load = self.f_y_middle
            else:
                load = self.u_y_middle
            if load is None:
                raise ValueError(f"No loading is given or defined before for the structure.")
        if loading_control.lower() == 'u':
            self.u_y_middle = load
            self.f_y_middle = None
        elif loading_control.lower() == 'f':
            self.f_y_middle = load
            self.u_y_middle = None
        else:
            raise ValueError(f"Loading control is not recognized. Possible values are 'u' and 'f' .")
        self.pars.loading_control = loading_control
    
    def refine_mesh(self, refinement_level):
        StructureFEniCS.refine_mesh(self, refinement_level) # this updates 'self.mesh_refinement_level'
        self.pars.resolutions['refinement_level'] = self.mesh_refinement_level
    
    def scale_mesh_resolutions(self, scale=1.0):
        self.pars.resolutions['res_y'] = round(scale * self.pars.resolutions['res_y'])
        self.mesh = self._build_mesh() # update the mesh based on scaled resolutions
    
    def increment_mesh_resolutions(self, increment=0):
        assert isinstance(increment, int)
        self.pars.resolutions['res_y'] = self.pars.resolutions['res_y'] + increment
        self.mesh = self._build_mesh() # update the mesh based on modified resolutions
    
    def is_inside(self, p):
        return ParsBend3Point2D.is_inside(pars=self.pars, p=p)
    
    def get_regular_grid_points(self, finement=2.0, x_range=None, include_edges=True):
        return ParsBend3Point2D.get_regular_grid_points(pars=self.pars, finement=finement, x_range=x_range, include_edges=include_edges)
    
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
        df.plot(self.mesh, color='gray', linewidth=min(1., rm/2.))
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