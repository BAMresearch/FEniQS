from feniQS.problem.QSM_GDM import *
from feniQS.problem.QSM_Elastic import *

def revise_pars_gdm(pars, c_min):
    pars.shF_degree_u = 2
    pars.integ_degree = 2
    pars.E_min = 40e3
    pars.E = 20e3 # E_total = 60e3
    pars.e0_min = 6e-4
    pars.e0 = 5e-4 # e0_total = 11e4
    pars.ef = 35e-4
    pars.c_min = c_min
    pars.c = 0. # c_total = 40.

def revise_pars_elastic_by_pars_gdm(pars_elastic, pars_gdm):
    pars_elastic.E = pars_gdm.E
    pars_elastic.E_min = pars_gdm.E_min
    pars_elastic.nu = pars_gdm.nu
    pars_elastic.constraint = pars_gdm.constraint

class AdmissibleElementalForces:
    def __init__(self, mesh, V_forces):
        """
        V_int_f: the space associated with nodal forces.
        """
        self.mesh = mesh
        self.V_forces = V_forces
    
    def _build_Ce_Me(self):
        pass

class LocalAssemblerPerCells:
    def __init__(self, mesh, form, V_form):
        """
        mesh: The mesh object over its cells the assembly is carried out.
        form: The form that is going to be assembled locally (per mesh nodes
            and per connected cells).
        V_form: The full function space that underlies tha given form.
        """
        self.mesh = mesh
        self.form = form
        self.V_form = V_form # must be a full space
        self._targets = dict()

        # Quantities per cell
        self.dofs_mapper = dict() # with keys being cell IDs
        self.assembles_per_cells = dict() # with keys being cell IDs
        for cell in df.cells(self.mesh):
            id_cell = cell.index()
            self.dofs_mapper[id_cell] = dict() # for each cell a dict with items associated to self._targets
            self.assembles_per_cells[id_cell] = None # will/can be updated through self.__call__.
    
    def add_target_space_and_points(self, label, V_target, points=None):
        """
        The self.mesh has a number of cells.
        The default value of 'points' is self.mesh.coordinates().
        This method assigns:
            self._targets[label] = {'points: points,
                                    'V': V_target,
                                    'V_dofs': V_dofs_target,
                                    'dofs_to_points_mapper': dofs_to_points_mapper}
            , where:
                'V_dofs_target' is the DOFs associated with 'V_target', but could be not over all the given points.
                'dofs_to_points_mapper' is a list of the length of 'V_dofs_target'
                , and dofs_to_points_mapper[m] = n means:
                    - If n is None: the m-th 'V_dofs_target' corresponds to no point.
                    - If n is an integer, the m-th 'V_dofs_target' corresponds to n-th point.
        and modifies:
            self.dofs_mapper, which is by itself is a dictionary with:
                keys: (unique) IDs of the cells of the self.mesh (as integers).
                values: for each cell, the value is a dictionary with:
                    keys: any label (of any target added now or before)
                    values: list of the same length as the number of LOCAL DOFs associated with
                        the 'self.V_form' and that particular cell. For example, if 'self.V_form' is
                        a function space with '2' subspaces and each cell has '3' nodes, then each value
                        of the dictionary is a list of length 6=3*2 .
                        And each entry of this list like 'value[i] = j' has the following meaning:
                        if 'j' is an integer:
                            then the i-th local DOF of that cell (as explained above) corresponds
                            to the 'j-th' V_dofs_target.
                        elif v is None:
                            then the i-th local DOF of that cell does not correspond to any 'V_dofs_target'.
        """
        if points is None:
            points = self.mesh.coordinates()
        V_dofs_target = V_target.dofmap().dofs()
        points_at_all_dofs_target = self.V_form.tabulate_dof_coordinates()[V_dofs_target,:]
        dofs_to_points_mapper = find_among_points(points=points_at_all_dofs_target, points0=points \
                                 , tol=self.mesh.rmin() / 1000.)
        self._targets[label] = {'points': points,
                                'V': V_target,
                                'V_dofs': V_dofs_target,
                                'dofs_to_points_mapper': dofs_to_points_mapper}
        for cell in df.cells(self.mesh):
            id_cell = cell.index()
            full_dofs_cell = self.V_form.dofmap().cell_dofs(id_cell)
            self.dofs_mapper[id_cell][label] = [] # initiation
            for full_dof in full_dofs_cell:
                try:
                    id_of_dof_target = V_dofs_target.index(full_dof)
                except ValueError:
                    id_of_dof_target = None
                self.dofs_mapper[id_cell][label].append(id_of_dof_target)
    
    def __call__(self):
        """
        It does the local assembly for all self.mesh.cells based on the current values
            that 'self.form' has adopted.
        """
        for cell in df.cells(self.mesh):
            self.assembles_per_cells[cell.index()] = df.assemble_local(self.form, cell)

    def collect_targets_per_points(self, reassemble=True):
        """
        It:
            1- (if reassemble==True):
                does the local assembly for all the cells (through self.__call__) based
                on the current values that 'self.form' has adopted.
            2- collects the assembled values corresponding to 'self._targets' (specified by
                the method 'self.add_target_space_and_points') per respective points.
        """
        if reassemble:
            self() # update 'self.assembles_per_cells'
        aa = {l: [[] for _ in range(len(target['points']))] for l, target in self._targets.items()}
        for cell in df.cells(self.mesh):
            id_cell = cell.index()
            full_assembles_at_cell = self.assembles_per_cells[id_cell]
            for l in self._targets.keys():
                try:
                    dofs_mapper_cell = self.dofs_mapper[id_cell][l]
                except KeyError:
                    raise KeyError(f"The target with label {l} is not resolved for the cell with ID={id_cell}.")
                for i, val in enumerate(full_assembles_at_cell):
                    form_dof = dofs_mapper_cell[i]
                    if form_dof is not None:
                        id_point = self._targets[l]['dofs_to_points_mapper'][form_dof]
                        if id_point is not None:
                            aa[l][id_point].append(val)
        return aa
    
    def collect_targets_per_cells(self, reassemble=True):
        """
        It:
            1- (if reassemble==True):
                does the local assembly for all the cells (through self.__call__) based
                on the current values that 'self.form' has adopted.
            2- collects the assembled values corresponding to 'self._targets' (specified by
                the method 'self.add_target_space_and_points') per respective cells.
            NOTE:
            This method will have a bug if points specified for each target do not contain the entire
                nodes of the mesh, because then some cells will miss some of their connected nodes.
            The static method 'build_full_assembler' is safe in that regard; i.e. it does NOT fall
                in such a bug.
        """
        if reassemble:
            self() # update 'self.assembles_per_cells'
        aa = {l: dict() for l in self._targets.keys()}
        for cell in df.cells(self.mesh):
            id_cell = cell.index()
            full_assembles_at_cell = self.assembles_per_cells[id_cell]
            for l in self._targets.keys():
                try:
                    dofs_mapper_cell = self.dofs_mapper[id_cell][l]
                except KeyError:
                    raise KeyError(f"The target with label {l} is not resolved for the cell with ID={id_cell}.")
                aa[l][id_cell] = []
                for i, val in enumerate(full_assembles_at_cell):
                    form_dof = dofs_mapper_cell[i]
                    if form_dof is not None:
                        aa[l][id_cell].append(val)
        return aa

    @staticmethod
    def build_full_assembler(fen, space='u', separate=False):
        """
        fen: an instance of 'FenicsProblem' class: the underlying FEniCS problem.
        This method builds (instantiates and adds proper targets to) a LocalAssemblerPerCells instance,
        such that for the space 'u' (displacements), the residuals of the underlying FEniCS problem
        (i.e. fen) will be locally assembled:
            - separately at each direction (potentially, 'fx', 'fy', 'fz'). <-- If separate==True
            - once in all directions. <-- If separate==False
        """
        assert isinstance(fen, FenicsProblem)
        F = fen.get_internal_forces()
        V = fen.get_i_full()
        if space.lower()=='u':
            iu = fen.get_iu()
        else:
            raise NotImplementedError()
        # Instantiate
        assembler = LocalAssemblerPerCells(fen.mesh, F, V)
        assembler.fen = fen # useful attribute
        # Add full targets
        if separate: # Separate per directions x,y,z
            num_sub_spaces = iu.num_sub_spaces()
            num_targets = max(1, num_sub_spaces)
            V_coords = V.tabulate_dof_coordinates()
            for _i, n in zip([0, 1, 2][:num_targets], ['fx', 'fy', 'fz'][:num_targets]):
                i = iu if (num_sub_spaces==0) else iu.sub(_i)
                i_dofs = i.dofmap().dofs()
                i_coords = V_coords[i_dofs]
                assembler.add_target_space_and_points(n, V_target=i, points=i_coords)
        else: # Once for all directions x,y,z
            u_dofs = iu.dofmap().dofs()
            u_coords = unique_points_at_dofs(V, u_dofs)
            assembler.add_target_space_and_points('f', V_target=iu, points=u_coords)
        return assembler

def get_num_DOFs_per_mesh_cell(V):
    return V.element().space_dimension()
    ## OR:
    # return len(V.dofmap().cell_dofs(0))

def get_nodes_of_space_at_cell(cell, V, V_full):
    return V_full.tabulate_dof_coordinates()[V.dofmap().cell_dofs(cell.index())]

def get_shF_degree(V):
    return V.ufl_element().degree()

def get_num_vertices_per_mesh_cell(cell):
    a = len(cell.get_vertex_coordinates())
    b = cell.mesh().geometric_dimension()
    if a%b==0:
        return a // b
    else:
        raise ValueError(f"Length of vertex coordinates (per cell) is not divisible by the cell geometric dimension.")

def get_integ_degree_and_num_GPs(dx, mesh):
    integ_degree = dx.metadata()['quadrature_degree']
    e_K = df.FiniteElement(family="Quadrature", cell=mesh.ufl_cell(), \
                            degree=integ_degree, quad_scheme="default")
    i_K = df.FunctionSpace(mesh, e_K)
    dim_per_cell = i_K.element().space_dimension()
    dim_total = i_K.dim()
    assert mesh.num_cells() * dim_per_cell == dim_total
    return integ_degree, dim_per_cell, dim_total

def fenics_statistics(fen):
    assert isinstance(fen, FenicsProblem)
    mesh = fen.mesh
    iu = fen.get_iu()
    iu0 = iu if iu.num_sub_spaces()==0 else iu.sub(0) # in one single direction
    geo_dim = mesh.geometric_dimension()
    if geo_dim == 1:
        n_eqs_per_node = 1
        n_eqs_per_cell = 1
        ss_dim = 1
        n_global_EQs = 1 # Number of balance EQs of the global structure.
    elif geo_dim == 2:
        n_eqs_per_node = 2
        n_eqs_per_cell = 3
        ss_dim = 3 # contributing to forces in 2D-plan
        n_global_EQs = 3 # Number of balance EQs of the global structure.
    elif geo_dim == 3:
        n_eqs_per_node = 3
        n_eqs_per_cell = 6
        ss_dim = 6
        n_global_EQs = 6 # Number of balance EQs of the global structure.
    # NOTE:
    # vertex: only concerns the mesh (regardless of iu0)
    # node  : related to iu0, thus, including potential interpolating nodes (for shF_degree_u larger than 1)
    num_cells = mesh.num_cells()

    num_vertices = mesh.num_vertices()
    num_vertices_per_cell = get_num_vertices_per_mesh_cell(df.Cell(mesh, 0))
    ave_num_cells_per_vertex = num_cells * num_vertices_per_cell / num_vertices

    shF_degree_iu = get_shF_degree(iu)
    num_nodes = iu0.dim()
    num_nodes_per_cell = get_num_DOFs_per_mesh_cell(iu0)
    ave_num_cells_per_nodes = num_cells * num_nodes_per_cell / num_nodes

    num_forces_per_point = max(1, iu.num_sub_spaces()) # genaral

    num_all_forces_structural = num_nodes * num_forces_per_point

    num_forces_per_cell = num_nodes_per_cell * num_forces_per_point
    num_indepAtCell_forces_per_cell = num_forces_per_cell - n_eqs_per_cell # per-cell-indep. (only w.r.t. the equilibrium of cell)
    num_all_forces = num_cells * num_forces_per_cell
    num_all_indepAtCell_forces = num_cells * num_indepAtCell_forces_per_cell
    num_redundant_EQs_nodes = num_nodes * n_eqs_per_node - n_global_EQs
    num_redundant_EQs_cells = num_cells * n_eqs_per_cell
    num_redundant_EQs = num_redundant_EQs_nodes + num_redundant_EQs_cells
    num_indep_forces = num_all_forces - num_redundant_EQs
    num_nonzero_coeffs_of_redundant_EQs_at_nodes = num_all_indepAtCell_forces * (1)
        # Each per-cell-indep. elemental force participates exactly in ONE EQ:
            # - translational balance of one node (which that elemental force corresponds to) in one spatial direction,
        # Other equations regarding the balance of cell; i.e. (1 + geo_dim - 1) are already considered in local-independence of the elemental forces.
    ratio_num_nonzero_coeffs_to_num_EQsCoeffsAtNodes = \
        num_nonzero_coeffs_of_redundant_EQs_at_nodes / (num_redundant_EQs_nodes * num_all_indepAtCell_forces)
    integ_degree, num_GPs_per_cell, num_GPs = get_integ_degree_and_num_GPs(fen.dxm, fen.mesh)
    num_ss_per_cell = num_GPs_per_cell * ss_dim
    num_stresses = num_GPs * ss_dim
    num_indep_stresses = num_stresses - num_redundant_EQs_nodes # 'num_redundant_EQs_cells' are already fulfilled when dealing with stresses

    print('GEOMETRY & CELLs -------------------------------------------------------------------------------')
    print(f"\tGeometric dimension:                                       {geo_dim}")
    print(f"\tTotal num. of cells:                                       {num_cells}")
    print('VERTICEs ---------------------------------------------------------------------------------------')
    print(f"\tTotal num. of vertices:                                    {num_vertices}")
    print(f"\tNum. of vertices per cell:                                 {num_vertices_per_cell}")
    print(f"\tAve. num. of cells connected to a vertex:                  {ave_num_cells_per_vertex:.2f}")
    print('NODEs (potentially containing interpolation nodes) ---------------------------------------------')
    print(f"\tInterpolation degree:                                      {shF_degree_iu}")
    print(f"\tTotal num. of nodes:                                       {num_nodes}")
    print(f"\tNum. of nodes per cell:                                    {num_nodes_per_cell}")
    print(f"\tAve. num. of cells connected to a node:                    {ave_num_cells_per_nodes:.2f}")
    print('EQUILIBRIUM ------------------------------------------------------------------------------------')
    print(f"\tNum. of equilibrium EQs. per node:                         {n_eqs_per_node}")
    print(f"\tNum. of equilibrium EQs. per cell:                         {n_eqs_per_cell}")
    print(f"\tNum. of equilibrium EQs. of the whole structure:           {n_global_EQs}")
    print('FORCEs (per point) -----------------------------------------------------------------------------')
    print(f"\tNum. of forces per point:                                  {num_forces_per_point}")
    print('NODAL FORCEs (STRUCTURAL)-----------------------------------------------------------------------')
    print(f"\tTotal num. of structural nodal forces (like in FEMU-F):    {num_all_forces_structural} = {num_nodes}*{num_forces_per_point}")
    print('NODAL FORCEs (ELEMENTAL) -----------------------------------------------------------------------')
    print(f"\tNum. of elemental (nodal) forces per cell:                 {num_forces_per_cell} = {num_nodes_per_cell}*{num_forces_per_point}")
    print(f"\tTotal num. of elemental (nodal) forces:                    {num_all_forces} = {num_cells}*{num_forces_per_cell}")
    print('NODAL FORCEs (ELEMENTAL) - per-CELL-INDEPENDENT ------------------------------------------------')
    print(f"\tNum. of per-cell-indep. elemental forces per cell:         {num_indepAtCell_forces_per_cell} = {num_forces_per_cell}-{n_eqs_per_cell}")
    print(f"\tTotal num. of per-cell-indep. elemental (nodal) forces:    {num_all_indepAtCell_forces} = {num_cells}*{num_indepAtCell_forces_per_cell}")
    print(f"\tNum. of EQs at cells among elemental forces:               {num_redundant_EQs_cells} = {num_cells}*{n_eqs_per_cell}")
    print('CONSTRAINTs (at nodes & among per-cell-indep. elemental forces) --------------------------------')
    print(f"\tNum. of EQs among elemental forces at nodes:               {num_redundant_EQs_nodes} = {num_nodes}*{n_eqs_per_node}-{n_global_EQs}")
    print(f"\tTotal num. of nonzero coeffs. of constraint EQs at nodes:  {num_nonzero_coeffs_of_redundant_EQs_at_nodes} = {num_all_indepAtCell_forces}*1")
    print(f"\tSparsity of nonzero entries of constraint EQs matrix:      {ratio_num_nonzero_coeffs_to_num_EQsCoeffsAtNodes:.1e} = {num_nonzero_coeffs_of_redundant_EQs_at_nodes}/({num_redundant_EQs_nodes}*{num_all_indepAtCell_forces})")
    print('NODAL FORCEs (ELEMENTAL) - FULLY-INDEPENDENT ---------------------------------------------------')
    print(f"\tTotal num. of redundancy EQs among elemental forces:       {num_redundant_EQs} = {num_redundant_EQs_nodes}+{num_redundant_EQs_cells}")
    print(f"\tTotal num. of fully-independent elemental nodal forces:    {num_indep_forces} = {num_all_indepAtCell_forces}-{num_redundant_EQs_nodes}={num_all_forces}-{num_redundant_EQs}")
    print(f"\t(1) Ratio of num. of fully-indep. nodal forces \
                \n\t\tto total num of per-cell-indep. nodal forces:      {(num_indep_forces / num_all_indepAtCell_forces):.3f} = {num_indep_forces}/{num_all_indepAtCell_forces}")
    print(f"\t(2) Ratio of num. of fully-indep. nodel forces \
                \n\t\tto num. of cells:                                  {(num_indep_forces / num_cells):.3f} = {num_indep_forces}/{num_cells}")
    print('GAUSS POINTs -----------------------------------------------------------------------------------')
    print(f"\tIntegration degree:                                        {integ_degree}")
    print(f"\tNum. of Gauss points per cell:                             {num_GPs_per_cell}")
    print(f"\tTotal num. of Gauss points:                                {num_GPs} = {num_cells}*{num_GPs_per_cell}")
    print('STRESS SPACE -----------------------------------------------------------------------------------')
    print(f"\tNum. of stress components per Gauss point:                 {ss_dim}")
    print(f"\tNum. of stress components per cell:                        {num_ss_per_cell} = {num_GPs_per_cell}*{ss_dim}")
    print(f"\tTotal num. of stress components at all GPs (like in mCRE): {num_stresses} = {num_GPs}*{ss_dim}")
    print(f"\tTotal num. of independent stress components:               {num_indep_stresses} = {num_stresses}-{num_redundant_EQs_nodes}")
    print(f"\t(1) Ratio of num. of fully-indep. stresses \
                \n\t\tto their total num:                                {(num_indep_stresses / num_stresses):.3f} = {num_indep_stresses}/{num_stresses}")
    print(f"\t(2) Ratio of num. of fully-indep. stresses \
                \n\t\tto num. of cells:                                  {(num_indep_stresses / num_cells):.3f} = {num_indep_stresses}/{num_cells}")
    print('SUMMARY ----------------------------------------------------------------------------------------')
    print(f"\tNum. of independent (forces, stresses):                    {num_indep_forces}, {num_indep_stresses}")
    print(f"\t(3) Length of constitutive errors in \
                \n\t\t(all forces, per-cell-indep. forces, stresses):    {num_all_forces}, {num_all_indepAtCell_forces}, {num_stresses}")
    print(f"\tNum. of (all elem. forces \
                \n\t\t,per-cell-indep. elem. forces, stresses) per cell: {num_forces_per_cell}, {num_indepAtCell_forces_per_cell}, {num_ss_per_cell}")
    print('------------------------------------------------------------------------------------------------')
    print(f"(1): Relevant for inference: the ratio of #unknowns to length of errors in constitutive law.")
    print(f"(2): Relevant for selecting indep. unknowns distributed over the cells.")
    print(f"(3): Relevant for inference: the length of errors in constitutive law.")

if __name__ == "__main__":
    df.set_log_level(30)
    
    ## GENERAL PARs
    c_min = 300.

    ## STRUCTURE's parameters

    from feniQS.structure.struct_kozicki2013 import *
    struct_name = 'kozicki2013'
    struct_cls = Kozicki2013
    pars_struct = ParsKozicki2013()
    pars_struct.left_sup_w = 10
    pars_struct.right_sup_w = 10
    pars_struct.loading_level *= 2. # larger loading, causing more damage
    reaction_places_1 = ['y_middle', 'y_left', 'y_right']
    reaction_places_2 = ['y_left', 'y_right']
    constraint = 'PLANE_STRAIN'

    # from feniQS.structure.struct_peerling1996 import *
    # struct_name = 'peerling1996'
    # struct_cls = Peerling1996
    # pars_struct = ParsPeerling()
    # reaction_places_1 = ['left', 'right']
    # reaction_places_2 = ['right']
    # constraint = 'UNIAXIAL'

    ## STRUCTUREs
    if struct_name=='kozicki2013':
        pars_struct.resolutions['scale'] = 1.
        pars_struct.resolutions['el_size_max'] = np.sqrt(c_min) / 1.5
    elif struct_name=='peerling1996':
        pars_struct._res = int(pars_struct.L / np.sqrt(c_min)) + 1
    checkpoints = [float(a) for a in (np.arange(1e-1, 1.0, 1e-1))] + [1.]
    struct1 = struct_cls(pars=pars_struct, _path=f"./STRUCTUREs/{struct_name}_uy/")
    struct2 = struct_cls(pars=pars_struct, _path=f"./STRUCTUREs/{struct_name}_ux/")
    if struct_name=='kozicki2013':
        ## (1) No concentrated force
        directions = []
        locations = []
        values = []
        ## (2) One concentrated force
        # directions = ['x']
        # locations = [[pars_struct.lx/2, pars_struct.ly/2]]
        # values = [50]
        ## (3) Two concentrated forces
        # directions = ['x', 'x']
        # locations = [[pars_struct.lx/3, pars_struct.ly/2]
        #             , [2*pars_struct.lx/3, pars_struct.ly/2]]
        # values = [50, -50.]
        for d, l, v in zip(directions, locations, values):
            for st in [struct1, struct2]:
                st.add_concentrated_force(direction=d, location=l, value=v, scale=1.)
    f_y_middle = PiecewiseLinearOverDatapoints(ts=checkpoints, vals=len(checkpoints)*[0.], _save=False)
    if struct_name=='kozicki2013':
        struct2.switch_loading_control('f', load=f_y_middle)
        load_interval = abs(pars_struct.x_from - pars_struct.x_to)

    ## MATERIALs' parameters
    pars_gdm = GDMPars(pars0=pars_struct)
    revise_pars_gdm(pars_gdm, c_min)
    pars_gdm.constraint = constraint
    pars_elastic = ElasticPars(pars0=pars_struct)
    revise_pars_elastic_by_pars_gdm(pars_elastic, pars_gdm)

    ## MODELs (quasi static)
    model_gdm = QSModelGDM(pars=pars_gdm, struct=struct1)
    model_elastic = QSModelElastic(pars=pars_elastic, struct=struct2)

    fenics_statistics(model_gdm.fen)

    ## LOCAL ASSEMBLERs
    separate = True # per spatial directions x, y
    assembler_gdm = LocalAssemblerPerCells.build_full_assembler(model_gdm.fen, separate=separate)
    assembler_elastic = LocalAssemblerPerCells.build_full_assembler(model_elastic.fen, separate=separate)

    ## SOLVE OPTIONs
    solver_options_1 = get_fenicsSolverOptions(lin_sol='direct') # regarding a single load-step
    so1 = QuasiStaticSolveOptions(solver_options_1) # regarding incremental solution
    so1.checkpoints = checkpoints
    so1.t_end = 1.
    so1.dt = 0.01
    so1.reaction_places = reaction_places_1
    
    solver_options_2 = get_fenicsSolverOptions(case='linear', lin_sol='iterative')
    solver_options_2['lin_sol_options']['method'] = 'cg'
    solver_options_2['lin_sol_options']['precon'] = 'default'
    solver_options_2['lin_sol_options']['tol_abs'] = 1e-10
    solver_options_2['lin_sol_options']['tol_rel'] = 1e-10
    so2 = QuasiStaticSolveOptions(solver_options=solver_options_2)
    so2.checkpoints = so1.checkpoints
    so2.t_end = so1.t_end
    so2.dt = so1.dt
    so2.reaction_places = reaction_places_2

    ## SOLVEs (for example)
    # model_gdm.solve(so1)
    # rfs_gdm = model_gdm.pps[0].plot_reaction_forces(tits=so1.reaction_places)
    # fs_top = [sum(f)/load_interval for f in model_gdm.pps[0].reaction_forces_checked[0]]
    # struct2.f_y_middle.vals = fs_top
    # model_elastic.solve(so2)
    # rfs_elastic = model_elastic.pps[0].plot_reaction_forces(tits=so2.reaction_places)

    ## STUDY: collect elemental forces for gdm problem and respective elastic problem with the same loading force
    def get_key(E, e0, ef):
        return f"E{E}e0{e0}ef{ef}"

    nodal_Fs_gdm = dict()
    nodal_Fs_elastic = dict()
    loaded_Fs = dict()
    # Es = [15000., 32000.]
    # e0s = [0.0004, 0.0006]
    # efs = [0.0030, 0.0043]
    Es = [15000.]
    e0s = [0.0004]
    efs = [0.0030]
    for E in Es:
        model_gdm.pars.E.assign(E)
        model_elastic.pars.E.assign(E)
        for e0 in e0s:
            model_gdm.pars.e0.assign(e0)
            for ef in efs:
                model_gdm.pars.ef.assign(ef)
                k = get_key(E, e0, ef)
                try:
                    model_gdm.solve(so1)
                    nodal_Fs_gdm[k] = assembler_gdm.collect_targets_per_points()

                    fs_top = [sum(f)/load_interval for f in model_gdm.pps[0].reaction_forces_checked[0]]
                    loaded_Fs[k] = fs_top[-1] # The local assemblies pertains to the last time-step, thus, last loading point
                    struct2.f_y_middle.vals = fs_top
                    model_elastic.solve(so2)
                    nodal_Fs_elastic[k] = assembler_elastic.collect_targets_per_points()
                except Exception:
                    pass
    
    ## Find a point over a node within the damage zone
    for c in model_gdm.fen.mesh.coordinates():
        if (0.48 * pars_struct.lx/2 <= c[0] <= 0.52 * pars_struct.lx/2) \
            and (0.48 * pars_struct.ly/2 <= c[1] <= 0.52 * pars_struct.ly/2):
            point = c
            break
    
    num_plots = 5
    for f_key in assembler_gdm._targets.keys():
        id_node_1 = find_among_points([point], assembler_gdm._targets[f_key]['points'], tol=1e-4)[0]
        id_node_2 = find_among_points([point], assembler_elastic._targets[f_key]['points'], tol=1e-4)[0]
        plt.figure()
        counter = 0
        for k, fs_gdm in nodal_Fs_gdm.items():
            if counter < num_plots:
                fs_node_gdm = fs_gdm[f_key][id_node_1]
                fs_elastic = nodal_Fs_elastic[k]
                fs_node_elastic = fs_elastic[f_key][id_node_2]
                plt.plot(fs_node_gdm, label=f"gdm-{k}")
                plt.plot(fs_node_elastic, label=f"elastic-{k}", linestyle='--')
                counter += 1
        plt.legend(loc='center left',  bbox_to_anchor=(1, 0.5))
        plt.title(f"Elemental forces at point {point}")
        plt.xlabel('cell (attached to the point)')
        plt.ylabel(f_key)
        plt.show()

        counter = 0
        for k, fs_gdm in nodal_Fs_gdm.items():
            if counter < num_plots:
                fs_node_gdm = fs_gdm[f_key][id_node_1]
                fs_elastic = nodal_Fs_elastic[k]
                fs_node_elastic = fs_elastic[f_key][id_node_2]
                load = loaded_Fs[k]
                factor = 1.
                # factor = 1. / load
                # ratio = factor * (np.array(fs_node_gdm) / np.array(fs_node_elastic))
                ratio = 'none'
                print(f"\nPars =\n\t{k}\nApplied load =\n\t{load}\nRatio of local forces=\n\t{ratio}")
                counter += 1

    ## CHECK STATIC EQUILIBRIUM
    tol = abs(fs_top[-1]) * 1e-8
    ids_rp = dict() # IDs of DOFs of reaction_places in DOFs of assembler
    for name, model, ass, so in zip(['gdm', 'elastic'],
                                     [model_gdm, model_elastic],
                                     [assembler_gdm, assembler_elastic],
                                     [so1, so2]):
        print(f"IDs of reaction DOFs of {name} problem in Points of assembler:")
        dofs_rp = model.struct.get_reaction_dofs(so.reaction_places, model.fen.get_iu())
        ids = dict()
        for i, k in enumerate(assembler_gdm._targets.keys()):
            ids_k = dict()
            dofs_ass = ass._targets[k]['V'].dofmap().dofs()
            points_ass = ass._targets[k]['dofs_to_points_mapper']
            for ip, rp in enumerate(so.reaction_places):
                dofs_ass_rp = [dofs_ass.index(d) for d in dofs_rp[ip] if d in dofs_ass]
                ids_k[rp] = [points_ass[m] for m in dofs_ass_rp]
            ids[k] = ids_k
            print(f"\t{k}: {ids_k}")
        ids_rp[name] = ids

    for name, ass in zip(['gdm', 'elastic'], [assembler_gdm, assembler_elastic]):
        print(f"\n\n-------- Check static balance for '{name}' problem --------\n")
        
        fs_per_cells = ass.collect_targets_per_cells()
        print(f"\n\tExceptions of static balance of cells: (direction, cell-ID, sum, forces):\n")
        for f_key in assembler_gdm._targets.keys():
            fs_key = fs_per_cells[f_key]
            for id_cell, vals in fs_key.items():
                if not abs(sum(vals))<tol:
                    print("\t\t", f"{f_key}, {id_cell}, {sum(vals)}\n\t\t\t", [f"{v:.2e}" for v in vals], '.')
        
        fs_per_points = ass.collect_targets_per_points(reassemble=False) # In above call of 'collect_targets_per_cells' we already did local assembly.
        print(f"\n\tExceptions of static balance of points: (direction, point-ID, sum, forces):\n")
        for f_key in assembler_gdm._targets.keys():
            print(f"\t\t---------- {f_key} ----------")
            fs_key = fs_per_points[f_key]
            fs_rp = {k: 0. for k in ids_rp[name][f_key].keys()}
            for i_p, vals in enumerate(fs_key):
                if not abs(sum(vals))<tol:
                    print("\t\t", f"{f_key}, {i_p}, {sum(vals)}\n\t\t\t", [f"{v:.2e}" for v in vals], '.')
                    for k, v in ids_rp[name][f_key].items():
                        if i_p in v:
                            fs_rp[k] += sum(vals)
            fs_key_sum = sum([sum(vs) for vs in fs_key])
            print(f"\t\t\tSum of all elemental nodal forces (at all individual nodes) =\n\t\t {fs_key_sum:.4e}")
            print(f"\t\t\tSum of elemental nodal forces (at individual nodes) per reaction place =\n\t\t {fs_rp}")
            print(f"\t\t\tSum = {(sum([f for f in fs_rp.values()])):.4e}")
        
        f_ext = df.assemble(ass.fen.get_external_forces()).get_local()
        print(f"\n\tSum of all external forces = {(sum(f_ext)):.4e}\n")
    
    df.set_log_level(20)
    print("\n----- DONE! -----")