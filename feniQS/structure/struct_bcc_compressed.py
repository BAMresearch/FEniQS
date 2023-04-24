import sys
if './' not in sys.path:
    sys.path.append('./')

from feniQS.structure.structures_fenics import *
from feniQS.fenics_helpers.fenics_functions import *
from feniQS.structure.helper_loadings import *
from feniQS.structure.helper_mesh_fenics import *

pth_struct_bcc_compressed = CollectPaths('struct_bcc_compressed.py')
pth_struct_bcc_compressed.add_script(pth_structures_fenics)
pth_struct_bcc_compressed.add_script(pth_fenics_functions)
pth_struct_bcc_compressed.add_script(pth_helper_loadings)
pth_struct_bcc_compressed.add_script(pth_helper_mesh_fenics)

class ParsBccCompressed(ParsBase):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            
            ## GEOMETRY
                # CASE (1): parametric (if no 'link_mesh_xdmf' is provided)
            self.shape_name   = "bcc"
            self.n_rve        = 2
            self.l_rve        = 10. / 3.
            self.r_strut      = 0.5
            self.add_plates   = True
            self.l_cell       = 0.2 # for mesh resolution
            
                # CASE (2): form a link (e.g. provided elsewhere)
            # self.link_mesh_xdmf = './DATAs/meshes/bcc/bcc_1/220214_test_A.xdmf'
            # Remove parameters if link_mesh_xdmf is given
            
            if hasattr(self, 'link_mesh_xdmf'):
                for k in ['shape_name', 'n_rve', 'l_rve', 'r_strut', 'add_plates', 'l_cell']:
                    delattr(self, k)
            
            ## LOADs and BCs
            self.loading_level = -0.3# A total level (magnifier) of loading (at middle)
            self.loading_scales = [1] # Several possible peaks for loading (at middle) --> peaks = level * scales
            self.loading_case = 'ramp' # or 'sin' or 'zigzag'
            self.loading_N = 1 # Number of periods for case='sin' or case='zigzag'
            self.loading_T = 1. # Maximum 't' of DEFINED loading
            self.loading_t_end = 1. # (=self.loading_T) Maximum 't' of APPLIED loading
            
            ## REST
            self._write_files = True
        else:
            ParsBase.__init__(self, **kwargs)

class BccCompressed(StructureFEniCS):
    def __init__(self, pars, _path=None, _name=None):
        if isinstance(pars, dict):
            pars = ParsBccCompressed(**pars)
        if _name is None:
            _name = 'BccCompressed'
            if hasattr(pars, 'link_mesh_xdmf'):
                _name += '_xdmfMesh'
            else:
                _name += '_parametricMesh'
        StructureFEniCS.__init__(self, pars, _path, _name)
    
    def _build_structure(self, _build_middle_load=True):
        
        if hasattr(self.pars, 'link_mesh_xdmf'):        
            ### IMPORT MESH ###
            self.mesh = df.Mesh()
            with df.XDMFFile(self.pars.link_mesh_xdmf) as f:
                f.read(self.mesh)
        else:
            self.mesh = bcc_mesh_parametric(r_strut      = self.pars.r_strut,
                                             l_rve       = self.pars.l_rve,
                                             n_rve       = self.pars.n_rve,
                                             l_cell      = self.pars.l_cell,
                                             add_plates  = self.pars.add_plates,
                                             shape_name  = self.pars.shape_name,
                                             _path       = self._path,
                                             _name       = self._name
                                             )
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)

        ### LOADs ###
        if _build_middle_load:
            self.u_z_top = time_varying_loading(fs=self.pars.loading_level*np.array(self.pars.loading_scales) \
                                                  , N=self.pars.loading_N, T=self.pars.loading_T, _case=self.pars.loading_case \
                                                  , _path=self._path)
    
    def get_BCs(self, i_u):
        time_varying_loadings = {'z_top': self.u_z_top}
        
        cs = self.mesh.coordinates()
        x_min, y_min, z_min = min(cs[:,0]), min(cs[:,1]), min(cs[:,2])
        x_max, y_max, z_max = max(cs[:,0]), max(cs[:,1]), max(cs[:,2])
        
        xy_max_dist = ((x_max-x_min) + (y_max-y_min))/2
        z_max_dist = z_max-z_min
        # tol_bc = self.mesh.rmin() # rmin() is too small to define tolerances for scanned meshes!
        z_tol = z_max_dist/500.
        xy_tol = xy_max_dist/25.
        
        ### HOMOGENEOUS DR BCs
        bcs_DR = {}
        
        def bottom_plate(x, on_boundary):
            return on_boundary and df.near(x[2], z_min, z_tol)
        
        ## CASE-1: fix bottom plate entirely in z-direction, and the edges in x and y directions
        bot_z, bot_z_dofs = boundary_condition(i_u.sub(2), df.Constant(0.0), bottom_plate)
        bcs_DR.update({'bot_z': {'bc': bot_z, 'bc_dofs': bot_z_dofs}})
        def bottom_x(x): # x-Bewegung = 0
            return (
                df.near(x[2], z_min, z_tol) and
                df.near(x[0], x_min, xy_tol))
        bot_x_ax, bot_x_ax_dofs = boundary_condition_pointwise(i_u.sub(0), df.Constant(0.0), bottom_x)
        bcs_DR.update({'bot_x_ax': {'bc': bot_x_ax, 'bc_dofs': bot_x_ax_dofs}})
        def bottom_y(x): # y-Bewegung = 0
            return (df.near(x[2], z_min, z_tol) and
                    df.near(x[1], y_min, xy_tol))
        bot_y_ax, bot_y_ax_dofs = boundary_condition_pointwise(i_u.sub(1), df.Constant(0.0), bottom_y)
        bcs_DR.update({'bot_y_ax': {'bc': bot_y_ax, 'bc_dofs': bot_y_ax_dofs}})
        
        ## CASE-2: fix bottom plate at all DOFs
        # bot, bot_dofs = boundary_condition(i_u, df.Constant((0., 0., 0.)), bottom_plate)
        # bcs_DR.update({'bot': {'bc': bot, 'bc_dofs': bot_dofs}})
        
        ### INHOMOGENEOUS DR BCs
        bcs_DR_inhom = {}
        
        def top_plate(x, on_boundary):
            return on_boundary and df.near(x[2], z_max, z_tol)
        top_z, top_z_dofs = boundary_condition(i_u.sub(2), self.u_z_top, top_plate)
        bcs_DR_inhom.update({'top_z': {'bc': top_z, 'bc_dofs': top_z_dofs}})
        
        ### Export marked BCs ### can be useful for investigation/ debugging
        # dataname = "./problems/Lattice_Mesh/facets.pvd"
        # facets = df.MeshFunction("size_t", self.mesh, 1)
        # df.AutoSubDomain(bottom_plate).mark(facets, 1)
        # df.AutoSubDomain(bottom_x).mark(facets, 2)
        # df.AutoSubDomain(bottom_y).mark(facets, 3)
        # df.AutoSubDomain(top_plate).mark(facets, 4)
        # df.File(dataname) << facets
        # time_varying_loadings ???
        
        return bcs_DR, bcs_DR_inhom, time_varying_loadings

if __name__=='__main__':
    pars = ParsBccCompressed()
    struct = BccCompressed(pars)
    struct.yamlDump_pars()
