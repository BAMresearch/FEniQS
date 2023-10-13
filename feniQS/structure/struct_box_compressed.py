from feniQS.structure.structures_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_loadings import *

pth_struct_box_compressed = CollectPaths('./feniQS/structure/struct_box_compressed.py')
pth_struct_box_compressed.add_script(pth_structures_fenics)
pth_struct_box_compressed.add_script(pth_fenics_functions)
pth_struct_box_compressed.add_script(pth_helper_loadings)

class ParsBoxCompressed(ParsBase):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            ## GEOMETRY
            self.lx = 10.
            self.ly = 10.
            self.lz = 10.
            
            ## MESH
            self.res_x = 10
            self.res_y = 10
            self.res_z = 10
            
            ## LOADs and BCs
            self.loading_level = -1.# A total level (magnifier) of loading (at middle)
            self.loading_scales = [1] # Several possible peaks for loading (at middle) --> peaks = level * scales
            self.loading_case = 'ramp' # or 'sin' or 'zigzag'
            self.loading_N = 1 # Number of periods for case='sin' or case='zigzag'
            self.loading_T = 1. # Maximum 't' of DEFINED loading
            self.loading_t_end = 1. # (=self.loading_T) Maximum 't' of APPLIED loading
            
            ## REST
            self._write_files = True
        else:
            ParsBase.__init__(self, **kwargs)

class BoxCompressed(StructureFEniCS):
    def __init__(self, pars, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsBoxCompressed(**pars)
        if _name is None:
            _name = 'BoxCompressed'
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_structure(self, _build_middle_load=True):
        ### MESH ###
        P1 = df.Point(0., 0., 0.)
        P2 = df.Point(self.pars.lx, self.pars.ly, self.pars.lz)
        self.mesh = df.BoxMesh(P1, P2, self.pars.res_x, self.pars.res_y, self.pars.res_z)
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
        
        ### LOADs ###
        self.u_z_top = None
        if _build_middle_load:
            self.u_z_top = time_varying_loading(fs=self.pars.loading_level*np.array(self.pars.loading_scales) \
                                                  , N=self.pars.loading_N, T=self.pars.loading_T, _case=self.pars.loading_case \
                                                  , _path=self._path)
    
    def get_time_varying_loadings(self):
        time_varying_loadings = {}
        if self.u_z_top is not None:
            time_varying_loadings['z_top'] = self.u_z_top
        return time_varying_loadings

    def build_BCs(self, i_u):
        cs = self.mesh.coordinates()
        x_min, y_min, z_min = min(cs[:,0]), min(cs[:,1]), min(cs[:,2])
        x_max, y_max, z_max = max(cs[:,0]), max(cs[:,1]), max(cs[:,2])
        
        tol_bc = self.mesh.rmin()/1000.
        
        ### HOMOGENEOUS DR BCs
        self.bcs_DR = {}
        
        def bottom_plate(x, on_boundary):
            return on_boundary and df.near(x[2], z_min, tol_bc)
        bot_z, bot_z_dofs = boundary_condition(i_u.sub(2), df.Constant(0.0), bottom_plate)
        self.bcs_DR.update({'bot_z': {'bc': bot_z, 'bc_dofs': bot_z_dofs}})
        
        def bottom_x(x): # x-Bewegung = 0
            return (
                df.near(x[2], z_min, tol_bc) and
                df.near(x[0], x_min, tol_bc))
        bot_x_ax, bot_x_ax_dofs = boundary_condition_pointwise(i_u.sub(0), df.Constant(0.0), bottom_x)
        self.bcs_DR.update({'bot_x_ax': {'bc': bot_x_ax, 'bc_dofs': bot_x_ax_dofs}})
        
        def bottom_y(x): # y-Bewegung = 0
            return (df.near(x[2], z_min, tol_bc) and
                    df.near(x[1], y_min, tol_bc))
        bot_y_ax, bot_y_ax_dofs = boundary_condition_pointwise(i_u.sub(1), df.Constant(0.0), bottom_y)
        self.bcs_DR.update({'bot_y_ax': {'bc': bot_y_ax, 'bc_dofs': bot_y_ax_dofs}})
        
        ### INHOMOGENEOUS DR BCs
        self.bcs_DR_inhom = {}
        
        def top_plate(x, on_boundary):
            return on_boundary and df.near(x[2], z_max, tol_bc)
        top_z, top_z_dofs = boundary_condition(i_u.sub(2), self.u_z_top, top_plate)
        self.bcs_DR.update({'top_z': {'bc': top_z, 'bc_dofs': top_z_dofs}})
