from feniQS.structure.structures_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_loadings import *

pth_struct_rectangle2d = CollectPaths('./feniQS/structure/struct_rectangle2D.py')
pth_struct_rectangle2d.add_script(pth_structures_fenics)
pth_struct_rectangle2d.add_script(pth_fenics_functions)
pth_struct_rectangle2d.add_script(pth_helper_loadings)

class ParsRectangle2D(ParsBase):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            ## GEOMETRY
            self.lx = 1.
            self.ly = 1.
            
            ## MESH
            self.res_x = 5
            self.res_y = 5
            
            ## LOADs and BCs
            self.loading_control = 'u'
            self.loading_level = self.ly / 100
            
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

class Rectangle2D(StructureFEniCS):
    def __init__(self, pars, embedded_nodes=[], _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsBase(**pars)
        if _name is None:
            _name = 'rectangle2d'
        self.embedded_nodes = embedded_nodes # Should be set before 'StructureFEniCS.__init__', since '_build_structure' is called within 'StructureFEniCS.__init__'.
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_structure(self, _build_load=True):
        ### MESH ###
        self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(self.pars.lx, self.pars.ly), self.pars.res_x, self.pars.res_y)
        if self.pars._plot:
            plt.figure()
            df.plot(self.mesh)
            if len(self.embedded_nodes) > 0:
                plt.plot(self.embedded_nodes[:,0], self.embedded_nodes[:,1], marker='.', fillstyle='none' \
                         , linestyle='', color='red', label='Embedded nodes')
                plt.legend()
            plt.title(f"Mesh of '{self._name}'")
            plt.savefig(self._path + 'meshView.png', bbox_inches='tight', dpi=300)
            plt.show()
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
        
        self.u_y_top = None # by default
        self.f_y_top = None # by default
        ### LOADs ###
        if _build_load:
            l = time_varying_loading(fs=self.pars.loading_level*np.array(self.pars.loading_scales) \
                                                      , N=self.pars.loading_N, T=self.pars.loading_T, _case=self.pars.loading_case \
                                                      , _plot=self.pars._plot, lab_y=self.pars.loading_control.lower(), _path=self._path)
            if self.pars.loading_control.lower() == 'u':
                self.u_y_top = l
            elif self.pars.loading_control.lower() == 'f':
                self.f_y_top = l
            else:
                raise ValueError(f"Loading control of the structure is not recognized.")
    
    def get_BCs(self, i_u):
        assert i_u.num_sub_spaces() == 2
        time_varying_loadings = {}
        if self.u_y_top is not None:
            time_varying_loadings.update({'y_top': self.u_y_top})
        if self.f_y_top is not None:
            time_varying_loadings.update({'y_top': self.f_y_top})
        
        tol = self.mesh.rmin() / 1000.
        def bot_left(x, on_boundary):
            return df.near(x[0], 0., tol) and df.near(x[1], 0., tol)
        def bot_right(x, on_boundary):
            return df.near(x[0], self.pars.lx, tol) and df.near(x[1], 0., tol)
        
        bcs_DR, bcs_DR_inhom = {}, {}
        
        bc_bl_x, bc_bl_x_dofs = boundary_condition_pointwise(i_u.sub(0), df.Constant(0.0), bot_left)
        bc_bl_y, bc_bl_y_dofs = boundary_condition_pointwise(i_u.sub(1), df.Constant(0.0), bot_left)
        bc_br_y, bc_br_y_dofs = boundary_condition_pointwise(i_u.sub(1), df.Constant(0.0), bot_right)
        bcs_DR.update({'bot_left_x': {'bc': bc_bl_x, 'bc_dofs': bc_bl_x_dofs}})
        bcs_DR.update({'bot_left_y': {'bc': bc_bl_y, 'bc_dofs': bc_bl_y_dofs}})
        bcs_DR.update({'bot_right_y': {'bc': bc_br_y, 'bc_dofs': bc_br_y_dofs}})
        
        if self.u_y_top is not None:
            def top_left(x, on_boundary):
                return df.near(x[0], 0., tol) and df.near(x[1], self.pars.ly, tol)
            def top_right(x, on_boundary):
                return df.near(x[0], self.pars.lx, tol) and df.near(x[1], self.pars.ly, tol)
            bc_tl_y, bc_tl_y_dofs = boundary_condition_pointwise(i_u.sub(1), self.u_y_top, top_left)
            bc_tr_y, bc_tr_y_dofs = boundary_condition_pointwise(i_u.sub(1), self.u_y_top, top_right)
            bcs_DR_inhom.update({'top_left_y': {'bc': bc_tl_y, 'bc_dofs': bc_tl_y_dofs}})
            bcs_DR_inhom.update({'top_right_y': {'bc': bc_tr_y, 'bc_dofs': bc_tr_y_dofs}})
        
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
            if 'bot' in rp.lower():
                yy = 0.
            elif 'top' in rp.lower():
                yy = self.pars.ly
            else:
                raise NameError('Reaction place is not recognized. Specify the hight (top or bot).')
            
            ps = [[xx, yy]]
            nodes.append(ps)
        return nodes # A list of lists each being nodes related to a reaction place
    
    def get_support_nodes(self):
        return np.array([[0., 0.], [self.pars.lx, 0.]])
    
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
            if self.u_y_top is None:
                load = self.f_y_top
            else:
                load = self.u_y_top
            if load is None:
                raise ValueError(f"No loading is given or defined before for the structure.")
        if loading_control.lower() == 'u':
            self.u_y_top = load
            self.f_y_top = None
        elif loading_control.lower() == 'f':
            self.f_y_top = load
            self.u_y_top = None
        else:
            raise ValueError(f"Loading control is not recognized. Possible values are 'u' and 'f' .")
        self.pars.loading_control = loading_control