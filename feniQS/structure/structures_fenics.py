from feniQS.problem.model_time_varying import *
from feniQS.general.general import *

pth_structures_fenics = CollectPaths('./feniQS/structure/structures_fenics.py')
pth_structures_fenics.add_script(pth_model_time_varying)

class StructureFEniCS():
    def __init__(self, pars, _path=None, _name="StructureFEniCS"):
        if isinstance(pars, dict):
            pars = ParsBase(**pars)
        if _path is None:
            _path = './STRUCTUREs/' + _name + '/'
        self.pars = pars
        self._path = _path
        self._name = _name
        make_path(self._path)
        
        self.mesh = None # must be generated through overwritten '_build_structure'
        self.mesh0 = None
            # A back-up of the initially built mesh, as the mesh might get refined.
            # This will be backed-up as soon as 'self.refine_mesh' is called for the very first time.
        self.mesh_refinement_level = 0 # By default no refinement is applied.
        
        self._build_structure()
        self.build_special_fenics_fields()

        self.bcs_built = False # Structure has no BC objects before any FEM problem (e.g. elasticity, plasticity, ect.) is defined.
        self.bcs_DR = {}
        self.bcs_DR_inhom = {}
    
        self.concentrated_forces = {'x': [], 'y': [], 'z': []} # The way it will be formulated in FenicsProblem requires them to be separated in three groups (per direction).
    
    def yamlDump_pars(self):
        yamlDump_pyObject_toDict(self.pars, self._path + self._name + '_parameters.yaml')
    
    def _build_structure(self):
        ### MESH ###
        ### LOADs ###
        print("WARNING: The base implementation of the method '_build_structure' is not overwritten.")
    
    def get_BCs(self, i_u, fresh=True):
        """
        i_u: A FEniCS function space (or sub-space) corresponding to desired BCs.
            The user must take care of passing a proper i_u according to the implementation of
                any certain structure in view of a specific problem such as structural mechanics.
            For example for a structural mechanics problem, i_u is the function-space of displacement field.
        Returns:
            self.bcs_hom: dictionary containing homogeneous BCs, with:
                key: corresponding place and direction; e.g. top_middle_y, bot_left_xy, edge1_xyz, etc.
                value: a second dictionary with items regarding these keys:
                    'bc': FEniCS BC object
                    'bc_dofs': corresponding DOFs
            self.bcs_inhom: just the same as bcs_hom only regarding inhomogeneius BCs
        'fresh':
            True: A fresh set of BCs are built and returned.
            False:
                - If the structure already has some BCs, they will be returned.
                - If the structure does not yet have BCs, a fresh set is built and returned.
        """
        if fresh or (not self.bcs_built):
            self.build_BCs(i_u=i_u)
            self.bcs_built = True
        return self.bcs_DR, self.bcs_DR_inhom
    
    def get_time_varying_loadings(self):
        print("WARNING: The base implementation of the method 'get_time_varying_loadings' is not overwritten.")
        return {}

    def build_BCs(self, i_u):
        """
        i_u: A FEniCS function space (or sub-space) corresponding to desired BCs.
            The user must take care of passing a proper i_u according to the implementation of
                any certain structure in view of a specific problem such as structural mechanics.
            For example for a structural mechanics problem, i_u is the function-space of displacement field.
        Builds fresh BC objects:
            self.bcs_hom: dictionary containing homogeneous BCs, with:
                key: corresponding place and direction; e.g. top_middle_y, bot_left_xy, edge1_xyz, etc.
                value: a second dictionary with items regarding these keys:
                    'bc': FEniCS BC object
                    'bc_dofs': corresponding DOFs
            self.bcs_inhom: just the same as bcs_hom only regarding inhomogeneius BCs
        """
        print("WARNING: The base implementation of the method 'build_BCs' is not overwritten.")
    
    def get_reaction_dofs(self, reaction_places, i_u):
        """
        i_u: Any FEniCS function space (or sub-space) for which we extract DOFs at reaction_places.
        reaction_places : A list of strings relating to sensor positions.
        Returns:
            A list of lists each being DOFs related to given reaction places.
        """
        bcs_DR, bcs_DR_inhom = self.get_BCs(i_u, fresh=False)
        penalty_features = self.get_penalty_features(i_u)
        dofs = []
        for rp in reaction_places:
            rp_ = rp.split('_')
            dofs_rp = None
            for k in bcs_DR.keys():
                if all([r in k for r in rp_]):
                    dofs_rp = bcs_DR[k]['bc_dofs']
                    break
            if dofs_rp is None:
                for k in bcs_DR_inhom.keys():
                    if all([r in k for r in rp_]):
                        dofs_rp = bcs_DR_inhom[k]['bc_dofs']
                        break
            if dofs_rp is None:
                for k in penalty_features.keys():
                    if all([r in k for r in rp_]):
                        dofs_rp = penalty_features[k]['dofs']
                        break
            if dofs_rp is None:
                _msg = f"Reaction place '{rp}' is recognized neither among Dirichlet boundary conditions "
                _msg += f"(obtained from 'self.get_BCs(i_u)' method)"
                _msg += f", nor among penalty features specifying nodal springs."
                raise KeyError(_msg)
            dofs.append(dofs_rp)
        
        return dofs # A list of lists each being DOFs related to a reaction place
    
    def get_coordinates(self, coordinate_places=None, resolution=None):
        """
            coordinate_places: A list of string objects, whose each entry specifies a place at which
                points coordinates are returned.
            resolution: An integer or float number that specifies the resolution of points, whose
                coordinates are returned.
            This method returns a dictionary with keys being coordinate_places, and each value being
                a list of the respective points coordinates.
        """
        if (coordinate_places is None) or (len(coordinate_places)==0):
            return {}
        else:
            raise NotImplementedError(f"Overwrite the method 'get_coordinates', handling the input 'coordinate_places':\n\t{coordinate_places}.")
    
    def build_special_fenics_fields(self):
        """
        A dictionary of particular FEniCS fields (e.g. functions/expressions) for specific purposes.
        For example, an expression denoting that some part of the domain is damaged.
        """
        self.special_fenics_fields = {} # by default is empty
    
    def get_tractions_and_dolfin_measures(self):
        """
        This concerns any possible Neumann BC relevant for the structure.
        If the structure has any traction forces, it must return:
            - traction forces themselves (in df.as_tensor format)
            - corresponding mesures (ds) for each traction force
        """
        return [], []
    
    def get_tractions_dofs(self, i_full, i_u):
        """
        This concerns any possible Neumann BC relevant for the structure.
        If the structure has any traction forces, it must return:
            - DOFs related to all of Neumann BCs
        """
        return []
    
    def get_concentrated_forces(self):
        return self.concentrated_forces
    
    def add_concentrated_force(self, direction, location, value, scale=1.):
        """
        direction: 'x' or 'y' or 'z'.
        location: spatial position (point) at which the force is applied.
        value: the value of the force, which can be either a scalar or df.Expression or df.UserExpression.
        scale: a scalar (float) that only scales the gien value.
        """
        assert direction.lower() in ['x', 'y', 'z']
        f = {'location': location, 'value': value, 'scale': scale}
        self.concentrated_forces[direction].append(f)
    
    def get_penalty_features(self, i_u):
        """
        This concerns the degrees of freedom at which a spring with stiffness of 'penalty weight'
        should be modelled.
        i_u: The function space corresponding to the displacements.
        Returns a dictionary whose each key goes for example to a certain spring, and each value
        by itself is a dictionary with the following items:
            'weight': The spring coefficient. So far, a constant uniform value is implemented.
            'dofs': The degrees of freedom at which a spring is put.
            'u0': The reference value of displacement w.r.t. which the spring acts.
        """
        return {}

    def refine_mesh(self, refinement_level):
        if not (isinstance(refinement_level, int) and refinement_level>=0):
            raise ValueError(f"The 'refinement_level' must be a non-negative integer.")
        if refinement_level==self.mesh_refinement_level:
            print(f"The mesh is already in refinement_level = {refinement_level} .")
        elif refinement_level>self.mesh_refinement_level: # refinement to a larger level (more refined)
            if self.mesh0 is None:
                self.mesh0 = self.mesh # Done ONLY once.
            nrf = refinement_level -  self.mesh_refinement_level # Number of refinement
            for i in range(nrf):
                self.mesh = df.refine(self.mesh)
        else: # refinement to a smaller level (more coarse)
            assert (self.mesh0 is not None)
            self.mesh = self.mesh0
            for i in range(refinement_level):
                self.mesh = df.refine(self.mesh)
        self.mesh_refinement_level = refinement_level
    
    def plot_mesh(self, ax=None, _save=True, _path=None):
        raise NotImplementedError(f"Overwrite a specific 'plot_mesh' routine for your structure.")
    
class StructureFromMesh(StructureFEniCS):
    def __init__(self, mesh, _path=None, _name=None):
        StructureFEniCS.__init__(self, pars=None, _path=_path, _name=_name)
            # The mesh is directly given as input, so, 'pars' is useless and thus set to None.
        self.mesh = mesh
    def _build_structure(self):
        pass # Mesh is already set.

class FieldOverStructure:
    def __init__(self, pars_struct, evaluator):
        """
        pars_struct : ParsBase
            The parameters defining the structure, namely its geometry.
        evaluator : callable
            The inputs to such callable are:
                x: a spatial coordinate
                pars: parameters defining a structure.
        Then, the 'evaluate' method of this class will get only one input 'x',
        for which the field will be evaluated according to the given 'pars_struct' and 'evaluator'.
        """
        assert isinstance(pars_struct, ParsBase)
        assert callable(evaluator)
        self.pars_struct = pars_struct
        self.evaluator = evaluator
    def evaluate(self, x):
        return self.evaluator(x, self.pars_struct)
