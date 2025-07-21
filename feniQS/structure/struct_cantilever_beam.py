from feniQS.structure.structures_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_loadings import *
from feniQS.structure.helper_BCs import *

pth_struct_cantbeam = CollectPaths('./feniQS/structure/struct_cantilever_beam.py')
pth_struct_cantbeam.add_script(pth_structures_fenics)
pth_struct_cantbeam.add_script(pth_fenics_functions)
pth_struct_cantbeam.add_script(pth_helper_loadings)
pth_struct_cantbeam.add_script(pth_helper_BCs)

class ParsCantileverBeam2D(ParsBase):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            ## GEOMETRY
            self.lx = 10.
            self.ly = 1.
            
            ## MESH
            self.res_x = 40
            self.res_y = 4
            
            ## LOADs and BCs
            self.loading_control = 'u'
            self.loading_level = self.lx / 50
            
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

class ParsCantileverBeam3D(ParsCantileverBeam2D):
    pass # to be developed

class CantileverBeam2D(StructureFEniCS):
    def __init__(self, pars, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsBase(**pars)
        if _name is None:
            _name = 'cantileverBeam2d'
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_structure(self):
        ### MESH ###
        self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(self.pars.lx, self.pars.ly), self.pars.res_x, self.pars.res_y)
        if self.pars._plot:
            plt.figure()
            df.plot(self.mesh)
            plt.title(f"Mesh of '{self._name}'")
            plt.savefig(self._path + 'meshView.png', bbox_inches='tight', dpi=300)
            plt.show()
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
        
        ### LOADs ###
        self.u_y_tip = None # by default
        self.f_y_tip = None # by default
        l = time_varying_loading(fs=self.pars.loading_level*np.array(self.pars.loading_scales) \
                                                    , N=self.pars.loading_N, T=self.pars.loading_T, _case=self.pars.loading_case \
                                                    , _plot=self.pars._plot, lab_y=self.pars.loading_control.lower(), _path=self._path)

        if self.pars.loading_control.lower() == 'u':
            self.u_y_tip = l
        elif self.pars.loading_control.lower() == 'f':
            self.f_y_tip = l
            raise NotImplementedError(f"The force-controlled loading is not implemented for 'CantileverBeam2D'.")
        else:
            raise ValueError(f"Loading control of the structure is not recognized.")
    
    def get_time_varying_loadings(self):
        time_varying_loadings = {}
        if self.u_y_tip is not None:
            time_varying_loadings['y_tip'] = self.u_y_tip
        elif self.f_y_tip is not None:
            time_varying_loadings['y_tip'] = self.f_y_tip
        return time_varying_loadings

    def build_BCs(self, i_u):
        assert i_u.num_sub_spaces() == 2
        tol = self.mesh.rmin() / 1000.
        self.bcs_DR, self.bcs_DR_inhom = {}, {}

        def left_edge(x, on_boundary):
            return on_boundary and df.near(x[0], 0., tol)
        bc_left_x, bc_left_x_dofs = boundary_condition(i_u.sub(0), df.Constant(0.0), left_edge) # fix in x-direction
        self.bcs_DR.update({'left_x': {'bc': bc_left_x, 'bc_dofs': bc_left_x_dofs}})
        
            ## (1) fully clamped
        bc_left_y, bc_left_y_dofs = boundary_condition(i_u.sub(1), df.Constant(0.0), left_edge) # fix in y-direction
        self.bcs_DR.update({'left_y': {'bc': bc_left_y, 'bc_dofs': bc_left_y_dofs}})
            ## (2) free in y-direction except one node
        # def left_bot(x, on_boundary):
        #     return df.near(x[0], 0., tol) and df.near(x[1], 0., tol)
        # bc_left_y, bc_left_y_dofs = boundary_condition_pointwise(i_u.sub(1), df.Constant(0.0), left_bot)
        # bcs_DR.update({'left_bot_y': {'bc': bc_left_y, 'bc_dofs': bc_left_y_dofs}})
        
        if self.u_y_tip is not None:
            ## load on the right-tip node
            def right_top(x, on_boundary):
                return df.near(x[0], self.pars.lx, tol) and df.near(x[1], self.pars.ly, tol)
            bc_right, bc_right_dofs = boundary_condition_pointwise(i_u.sub(1), self.u_y_tip, right_top)
            self.bcs_DR_inhom.update({'right_tip_y': {'bc': bc_right, 'bc_dofs': bc_right_dofs}})
    
    def get_reaction_nodes(self, reaction_places):
        nodes = []
        tol = self.mesh.rmin() / 1000.
        for rp in reaction_places:
            rpl = rp.lower()
            if 'left' in rpl:
                if 'bot' in rpl:
                    ps = [[0., 0.]]
                elif 'top' in rpl:
                    ps = [[0., self.pars.ly]]
                else: # over the whole edge
                    ps = [pp for pp in self.mesh.coordinates() if abs(pp[0])<=tol]
            elif 'right' in rpl: # only concerns the tip (right top node))
                ps = [[self.pars.lx, self.pars.ly]]
            else:
                raise NameError('Reaction place is not recognized. Specify at least the side (left or right).')
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
            if self.u_y_tip is None:
                load = self.f_y_tip
            else:
                load = self.u_y_tip
            if load is None:
                raise ValueError(f"No loading is given or defined before for the structure.")
        if loading_control.lower() == 'u':
            self.u_y_tip = load
            self.f_y_tip = None
        elif loading_control.lower() == 'f':
            self.f_y_tip = load
            self.u_y_tip = None
        else:
            raise ValueError(f"Loading control is not recognized. Possible values are 'u' and 'f' .")
        self.pars.loading_control = loading_control

class CantileverBeam3D(CantileverBeam2D): ### ????
    pass # to be developed