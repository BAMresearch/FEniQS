import sys
if './' not in sys.path:
    sys.path.append('./')

import numpy as np
import dolfin as df
from feniQS.general.general import *

pth_fenics_functions = CollectPaths('fenics_functions.py')
pth_fenics_functions.add_script(pth_general)

def conditional_by_ufl(condition, f1, f2, value_true, value_false, tol=None):
    """
    condition: a string among the two following:
        'lt': less than (strictly)
        'le': less or equal
    NOTE: The other two possiblities ('gt': greater than (strictly) and 'ge': greater or equal)
            are simply achieved by switching the input functions f1 and f2.
    
    f1, f2: functions to be compared; i.e. it will be checked whether:
        $f1$ $condition$ $f2$
    
    value_true, value_false: respective return value depending on whether the condition is true or not.
    
    tol (optional): a POSITIVE tolerance for comparison, which is added to the less (smaller) function 'f1'.
    """
    try:
        ufl_condition = {'lt': df.lt, 'le': df.le}[condition]
    except KeyError:
        raise KeyError(f"Possible values for condition are 'lt' and 'le'. Consider switching input functions when intending the conditions 'gt' and 'ge'.")
    _f1 = f1
    if tol is not None:
        _f1 = f1 + df.Constant(tol)
    return df.conditional(ufl_condition(_f1, f2), value_true, value_false)


def get_element_volumes(mesh):
     V = df.FunctionSpace(mesh, 'DG', 0)
     u = df.Function(V)
     u.vector()[:] = 1.
     v = df.TestFunction(V)
     vol_form = v * u * df.dx(mesh)
     return df.assemble(vol_form).get_local()

def feed_nodal_disps_by_solving(mesh, points, values, xdmf_file=None \
                                , eval_points=[], _label='u', el_family='CG', degree=1, _remove=True):
    """
    points:
        MUST be a subset of mesh nodes (coordinates). ===> 'sparse nodes'
    values:
        A 3-dimensional array: n_ch * n_points * u_dim, where:
            - the first dimension goes to several checkpoints (load-steps),
            - the second dimension goes to sparce nodes, thus, MUST be sorted corresponding to the given points.,
            - the third dimension goes to the nomber of DOFs per each point/node (either of 1, 2, 3).
    xdmf_file:
        The full name of *.xdmf file that will be written.
    
    It first feeds values (at the points) as Dirichlet Boundary conditions
        and then solves a structural mechanics problem for the whole mesh to
        obtain values at other mesh nodes.
    """
    assert len(values.shape)==3
    assert values.shape[1]==points.shape[0]
    dim = values.shape[2]
    if points.shape[1] < mesh.geometric_dimension():
        # This e.g. emerges when reading an interval XDMF Mesh (in mesh_of_data):
            # mesh_of_data.geometric_dimension()=2, whereas eval_points is very likely an N*1 array.
        print("WARNING: The given points have missing dimensional coordinate(s) compared to the geometric dimension of the mesh. The missing coordinates will set to zero by default.")
        aa = mesh.geometric_dimension() - points.shape[1]
        points = np.concatenate(tuple([points] + [np.zeros_like(points)] * aa), axis=1)
    if dim==1:
        elem = df.FiniteElement(el_family, mesh.ufl_cell(), degree)
    else:
        elem = df.VectorElement(el_family, mesh.ufl_cell(), degree, dim=dim)
    V = df.FunctionSpace(mesh, elem)
    u = df.Function(V)
    u.rename(_label, _label)
    v = df.TestFunction(V)
    sig_u = 1.0 * df.grad(u) # assuming a dummy elastic modulus
    eps_v = df.grad(v)
    if dim>1:
        sig_u = df.sym(sig_u)
        eps_v = df.sym(eps_v)
    R = df.inner(sig_u, eps_v) * df.dx(mesh)
    dR = df.derivative(R, u)
    many_bc = ManyBCs(V=V, v_bc=V, points=points, sub_ids=[])
    ll = df.get_log_level()
    df.set_log_level(30) # To avoid console messages
    if xdmf_file is not None:
        if os.path.isfile(xdmf_file) and _remove:
            os.remove(xdmf_file)
    evals = []
    for i_ch, u_ch in enumerate(values):
        many_bc.assign(u_ch)
        df.solve(dR==R, u, bcs=[many_bc.bc])
        evals.append( [u(pp) for pp in eval_points] )
        if xdmf_file is not None:
            with df.XDMFFile(xdmf_file) as ff:
                ff.write_checkpoint(u, _label, i_ch + 1, append=True) # For (i_ch+1)-th load-step, with 1-based indexing.
    df.set_log_level(ll) # Back to default
    return np.array(evals)

def are_points_in_mesh(mesh, points):
    existing_points_ids = []
    missing_points_ids = []
    for i,p in enumerate(points):
        if is_point_in_mesh(mesh, p):
            existing_points_ids.append(i)
        else:
            missing_points_ids.append(i)
    return existing_points_ids, missing_points_ids

def is_point_in_mesh(mesh, point):
    _is = False; ic = 0
    while (not _is) and ic < mesh.num_cells():
        c = df.Cell(mesh, ic)
        _is = (c.collides(df.Point(point)) or c.contains(df.Point(point)))
        ic += 1
    return _is

def interpolate_data_of_meshPoints(mesh_of_data, data_at_mesh_points, eval_points \
                                   , el_family='CG', deg=1):
    """
    IMPORTANT:
        - 'data_at_mesh_points' MUST be sorted in the same order as mesh_of_data.coordiantes(),
        - This does NOT solve a structural mechanics problem, rather only interpolates a function
            with given values 'data_at_mesh_points'. Thus, if 'eval_points' happen to locate exterior
            to enough data, the interpolation will be based on default ZERO values of FEniCS function.
    Returns interpolation of given data at eval_points.
    """
    data_interpolated = []
    if len(data_at_mesh_points.shape) != 3:
        raise('The data_at_mesh_points should be provided in a 3-dimensional array: (#checkpoints, #mesh points, #DOFs/Values per point).')
    sh1, sh2, sh3 = data_at_mesh_points.shape # sh1: #checkpoints  ,  sh2: #data points  ,  sh3: #DOFs per point
    assert sh2 == mesh_of_data.num_vertices()
    if sh3==1:
        e1 = df.FiniteElement(el_family, mesh_of_data.ufl_cell(), deg)
    else:
        e1 = df.VectorElement(el_family, mesh_of_data.ufl_cell(), deg, dim=sh3)
    V1 = df.FunctionSpace(mesh_of_data, e1)
    f1 = df.Function(V1)
    data_dofs = []
    for i in range(sh3):
        ii = V1 if sh3==1 else V1.sub(i)
        data_dofs.append( dofs_at(mesh_of_data.coordinates(), V1, ii, mesh_of_data.rmin() / 1000.) )
    
    if eval_points.shape[1] < mesh_of_data.geometric_dimension():
        # This e.g. emerges when reading an interval XDMF Mesh (in mesh_of_data):
            # mesh_of_data.geometric_dimension()=2, whereas eval_points is very likely an N*1 array.
        print("WARNING: The given evaluation points have missing dimensional coordinate(s) compared to the geometric dimension of the mesh. The missing coordinates will set to zero by default.")
        aa = mesh_of_data.geometric_dimension() - eval_points.shape[1]
        eval_points = np.concatenate(tuple([eval_points] + [np.zeros_like(eval_points)] * aa), axis=1)
        
    for d in data_at_mesh_points: # per checkpoint
        # Assign data to f1
        for _id, dd in enumerate(d): # per point
            for _idir, ddd in enumerate(dd): # per direction
                f1.vector()[data_dofs[_idir][_id]] = ddd
        # Interpolate (simply evaluation of function)
        d_interpolated = [f1(pp) for pp in eval_points]
        # Append
        data_interpolated.append(d_interpolated)
    return np.atleast_3d(data_interpolated)

def subMesh_over_nodes(parent_mesh, nodes, points_tol=None):
    """
    This returns:
        - a submesh whose nodes will be for sure a subset of given nodes.
        - Possibly some non-contributing alone_nodes, which are NOT part of subMesh
            , since they could not belong to any cell whose all nodes are in the given nodes.
    """
    _dim = parent_mesh.geometric_dimension()
    n_v = parent_mesh.cells().shape[1] # number of vertices per cell
    cells_ids = []
    mf = df.MeshFunction('size_t', parent_mesh, _dim)
    mf.set_all(0)
    for ic in range(parent_mesh.num_cells()):
        c = df.Cell(parent_mesh, ic)
        nps = 0
        for ip in range(nodes.shape[0]): # loop over all given nodes
            pe = nodes[ip,:]
            if c.collides(df.Point(pe)) or c.contains(df.Point(pe)):
                nps+=1 # The cell has found a node among nodes.
            if nps==n_v: # The cell has got 'enough' nodes; i.e. the number of nodes per cell.
                # we consider the cell, since all of its nodes are among the given nodes.
                cells_ids.append(ic)
                mf.set_value(ic, 1)
                break
    mesh_new = df.SubMesh(parent_mesh, mf, 1)
    mesh_nodes = mesh_new.coordinates()
    alone_nodes = []
    if points_tol is None:
        points_tol = mesh_new.rmin() / 1000.
    for nc in nodes:
        _found = False
        for nn in mesh_nodes:
            if np.linalg.norm(nn-nc)<points_tol:
                _found=True
                break
        if not _found:
            alone_nodes.append(nc)
    return mesh_new, np.array(alone_nodes)

def dofmap_to_space_in_parent_mesh(Vsub, Vparent, tol=None):
    """
    Inspired from:
        https://fenicsproject.org/qa/8522/mapping-degrees-of-freedom-between-related-meshes/
        
    Vsub and Vparent must have the same DOFs (per node) but over different meshes.
    The mesh of Vsub must be a SubMesh of the mesh behind Vparent, which also
        implies that the nodes of the former should coincide the latter.
    It returns mapper such that:
        mapper[sub_dof] = parent_dof   (both global DOFs)
        ; where sub_dof goes to Vsub (in subMesh) and parent_dofs goes to Vparent.
    """
    def assert_matching_spaces(V1, V2, b=True):
        b = b and (V1.num_sub_spaces() == V2.num_sub_spaces())
        if b:
            for i in range(V1.num_sub_spaces()):
                b = assert_matching_spaces(V1.sub(i), V2.sub(i), b)
        return b
    b = assert_matching_spaces(Vsub, Vparent)
    if not b:
        raise ValueError('The function spaces not matching.')
    
    V_coords = Vparent.tabulate_dof_coordinates() # of the full dofs over parent mesh
    Vsub_coords = Vsub.tabulate_dof_coordinates() # of the full dofs over sub mesh
    if tol is None:
        tol = Vparent.mesh().rmin() / 1000.
    mapper = {}
    def doit_for_leave_spaces(Vs, Vp):
        if Vs.num_sub_spaces()==0:
            ## There is no more sub-space, so we do the main task.
            Vs_dofs_leave = Vs.dofmap().dofs()
            Vs_coords_leave = Vsub_coords[Vs_dofs_leave]
            Vp_dofs_leave = Vp.dofmap().dofs()
            Vp_coords_leave = V_coords[Vp_dofs_leave]
            for _id, sub_coords in enumerate(Vs_coords_leave):
                corresponding_ids = [i for i, coords in enumerate(Vp_coords_leave) if np.linalg.norm(coords-sub_coords)<tol]
                if len(corresponding_ids) == 1:
                    mapper[Vs_dofs_leave[_id]] = Vp_dofs_leave[corresponding_ids[0]]
                else:
                    raise ValueError("Degrees of freedom not matching.")
        else:
            for i in range(Vsub.num_sub_spaces()):
                Vss = Vs.sub(i)
                Vps = Vp.sub(i)
                doit_for_leave_spaces(Vss, Vps)
    
    doit_for_leave_spaces(Vsub, Vparent)
    return mapper

def get_contributing_points(contributed_points, mesh, zero_thr=1e-10, interpolation_thr=1e-5, points_tol=None):
    """
    The contributed_points MUST locate within the domain of mesh. Otherwise, get_contributing_mesh_features raises errors.
    - zero_thr: A threshold below which we consider NO contribution.
    - interpolation_thr: A threshold below which we neglect contribution.
    It returns:
    - all_contr_nodes: all of those nodes of mesh that are contributing to contributed_points (regarding interpolation), regardless of interpolation_thr.
    - contr_nodes: all_contr_nodes except negligible nodes that contribute too little (below interpolation_thr) to any contributed_points.
    """
    assert (zero_thr <= interpolation_thr)
    elem = df.FiniteElement('CG', mesh.ufl_cell(), 1)
    ii = df.FunctionSpace(mesh, elem)
    if points_tol is None:
        points_tol = mesh.rmin() / 1000.
    all_contr_dofs, all_contr_nodes, contr_dofs, contr_nodes, neglected_nodes, M = get_contributing_mesh_features(contributed_points, V_full=ii, V=ii, points_tol=points_tol \
                                                                , zero_thr=zero_thr, interpolation_thr=interpolation_thr, return_sparse=False)
    return np.array(all_contr_nodes), np.array(contr_nodes), np.array(neglected_nodes)

def get_contributing_mesh_features(contributed_points, V_full, V, points_tol=None \
                                   , zero_thr=1e-10, interpolation_thr=1e-5, return_sparse=False):
    """
    "Vfull": a full function space having "tabulate_dof_coordinates" attribute.
    "V" can be any space function. In case of being a sub-space, it must NOT be collapsed.
    "zero_thr": threshold of interpolation weight below which we consider NO contribution.
    "interpolation_thr": A threshold below which we neglect contribution.
    It returns:
        all contributing dofs
        all contributing nodes
        contributing dofs
        contributing nodes
        M: Full interpolation Matrix fpr the given contributed_points w.r.t. the given "V" space (regardless of zero_thr and interpolation_thr).
    """
    assert (zero_thr <= interpolation_thr)
    if points_tol is None:
        points_tol = mesh.rmin() / 1000.
    M = build_interpolation_matrix(xs=contributed_points, V=V, _return_sparse=return_sparse)
    # We detect contributing DOFs (to measured data), which correspond to columns of M that are NOT entirely ZERO.
    Vdofs = V.dofmap().dofs()
    all_contr_dofs = []; contr_dofs = []
    for j in range(M.shape[1]): # columns
        if any(abs(M[:,j])>zero_thr):
            contr_d = Vdofs[j]
            if contr_d not in all_contr_dofs:
                all_contr_dofs.append(contr_d)
            if any(M[:,j]>interpolation_thr) and (contr_d not in contr_dofs):
                contr_dofs.append(contr_d)
    ps = V_full.tabulate_dof_coordinates()[all_contr_dofs,:] # might be repeated
    all_contr_nodes = add_unique_points(ps, points0=[], _eps=points_tol)[0]
    ps = V_full.tabulate_dof_coordinates()[contr_dofs,:] # might be repeated
    contr_nodes = add_unique_points(ps, points0=[], _eps=points_tol)[0]
    neglected_nodes = []
    for pp in all_contr_nodes:
        _is = False
        for p in contr_nodes:
            if np.linalg.norm(pp - p) < points_tol:
                _is = True
                break
        if not _is:
            neglected_nodes.append(pp)
    assert (len(all_contr_nodes) == len(contr_nodes) + len(neglected_nodes))
    return all_contr_dofs, all_contr_nodes, contr_dofs, contr_nodes, neglected_nodes, M

def build_interpolation_matrix(xs, V, _return_sparse=True):
    """
    "V" can be any space function. In case of being a sub-space, it must NOT be collapsed.
    "xs" is a 2-D array containing coordinates of a number of arbitrary points
        , over which we compute the interpolation matrix M.
        These points must be within the domain of the mesh that is behind V.
    Return "M" matrix with the shape of (nx*nss, V.dim()), with:
        nx : number of given points
        nss: number of subspaces of V, i.e. number of DOFs per each point.
    The vector that should be multiplied by M is:
        - u_mix.split(deepcopy=False)[0].vector()[V.dofmap().dofs()]
            ; for a mixed problem with V=i_mix.sub(0)
        - u.vector().get_local()
            ; for a normal (no mixed) problem
    Inspired from:
        https://fenicsproject.org/qa/4372/interpolator-as-matrix/
    """
    nx = xs.shape[0]
    dim = V.dim()
    mesh = V.mesh()
    assert mesh.geometry().dim()==xs.shape[1]
    coords = mesh.coordinates()
    cells = mesh.cells()
    dolfin_element = V.dolfin_element()
    dofmap = V.dofmap()
    Vdofs = dofmap.dofs()
    nss = max(V.num_sub_spaces(), 1)
    bbt = mesh.bounding_box_tree()
    sdim = dolfin_element.space_dimension()
    rows = np.zeros(nx*nss*sdim, dtype='int')
    cols = np.zeros(nx*nss*sdim, dtype='int')  
    vals = np.zeros(nx*nss*sdim)
    for k in range(nx):
        x = list(xs[k,:]) # This "list" is unclearly needed to some weird issues in some cases - believe me :)
        # Find cell for the point
        try:
            cell_id = bbt.compute_first_entity_collision(df.Point(x))
        except:
            raise ValueError(f"The point {x} is not belonging to any mesh cell.")
        if cell_id>cells.shape[0]:
            raise ValueError(f"The cell detected to contain point {x} has an invalid ID of {cell_id} that is larger than number of cells ({cells.shape[0]}).")
        nodal_dofs = dofmap.cell_dofs(cell_id)
        # Vertex coordinates for the cell
        xvert = coords[cells[cell_id,:],:]
        # Evaluate the basis functions for the cell at x
        v = dolfin_element.evaluate_basis_all(x,xvert,cell_id)
        if not all([abs(vi)<1.0+1e-10 for vi in v]):
            raise ValueError("A shape function evaluation exceeds 1.0 .")
        _shift = k*nss*sdim
        for iss in range(nss): # per sub-space
            ij_iss = np.arange(sdim*iss,sdim*(iss+1))
            rows[ij_iss + _shift] = nss*k + iss
            cols[ij_iss + _shift] = np.array([Vdofs.index(d) for d in nodal_dofs])
            vals[ij_iss + _shift] = v[iss::nss]
    if _return_sparse:
        import scipy.sparse as sps
        ij = np.concatenate((np.array([rows]), np.array([cols])),axis=0)
        M = sps.csr_matrix((vals, ij), shape=(nx*nss,V.dim()))
    else:
        M = np.zeros((nx*nss,V.dim()))
        for i,j,val in zip(rows,cols,vals):
            M[i,j]=val
    return M

class DolfinFunctionsFeeder:
    def __init__(self, funcs, dofs):
        assert all([isinstance(ff, df.Function) for ff in funcs])
        self.funcs = funcs
        self.dofs = dofs
    def __call__(self, vals):
        assert len(vals) == len(self.dofs)
        for f in self.funcs:
            f.vector()[self.dofs] = vals[:]

class DolfinFunctionAssigner:
    """
    An instance of this helper class has a callable (self.__call__) by which we can assign a new value (val) to a FEniCS function/Constant (func).
    """
    def __init__(self, func):
        assert (type(func)==df.Constant or type(func)==df.Function)
        self.func = func
    def __call__(self, val): # as assigner
        self.func.assign(val)

class ManyBCs:
    def __init__(self, V, v_bc, points, sub_ids=[0], vals=None):
        """
        V :
            A full function space (not a subspace) used for representing the domain's coordinates.
        v_bc :
            The FEniCS function space (or subspace) over which BCs are generated. It can have several dimensions (2D, 3D).
        points : 
            List of np.array each being the coordinates of points on which the Dirichlet BCs are applied.
        sub_ids :
            A list of indices which is used to extract the sub-function of self.u_bc=df.Function(V) that is corresponding to "v_bc".
        vals (optinal):
            List of np.array each being the value of BC at the corresponding point: each entry might be a scalar or an np.array depending on _dim=v_bc.num_sub_spaces()
        self.bc:
            A single DirichletBC object representing all desired BCs over points with the given vals.
        """
        tol = V.mesh().rmin() / 1000.
        self._dim = v_bc.num_sub_spaces()
        if self._dim == 0 or self._dim == 1: # might also be zero for a scalar function_space.
            self._dim = 1
            vals0 = df.Constant(0.0)
        else:
            vals0 = df.Constant(self._dim * (0.0, ))
        dom = points_at(points, _eps=tol)
        self.bc = df.DirichletBC(v_bc, vals0, dom, method="pointwise")
        
        ## identify DOFs
        self.dofs = list(self.bc.get_boundary_values().keys())
        ## identify COORDs
        self.dofs_to_coords = V.tabulate_dof_coordinates()
        ## define/initiate FEniCS function representing BC
        self.u_bc = df.Function(V) # it MUST live on the full mixed-space
        self.u_bc_dofs = V.dofmap().dofs()
        self.sub_ids = sub_ids
        
        ## set self.mapper
        self.mapper = len(points) * [None]
        if self._dim == 1:
            coords_bc = self.dofs_to_coords[self.dofs]
            dofs_map = find_mixup(points, coords_bc, tol=tol)
            if not all(d!=-1 for d in dofs_map):
                ## in view of the implementation of the method "find_mixup"
                raise ValueError('At least one of the points does not match any node of the mesh corresponding to the specified function-space.')
            aa = [self.u_bc_dofs.index(d) for d in self.dofs] # indices of self.dofs in the dofs of u_bc (in mixed space) : self.dofs[i] = u_bc_dofs[aa[i]]
            for i, j in enumerate(dofs_map):
                self.mapper[j] = aa[i]
        elif self._dim == 2:
            dofs_x = v_bc.sub(0).dofmap().dofs()
            dofs_y = v_bc.sub(1).dofmap().dofs()
            bc_dofs_x = [d for d in self.dofs if d in dofs_x]
            bc_dofs_y = [d for d in self.dofs if d in dofs_y]
            coords_bc_x = self.dofs_to_coords[bc_dofs_x]
            coords_bc_y = self.dofs_to_coords[bc_dofs_y]
            dofs_map_x = find_mixup(points, coords_bc_x, tol=tol)
            dofs_map_y = find_mixup(points, coords_bc_y, tol=tol)
            if not all(d!=-1 for d in dofs_map_x+dofs_map_y):
                ## in view of the implementation of the method "find_mixup"
                raise ValueError('At least one of the points does not match any node of the mesh corresponding to the specified function-space.')
            aa_x = [self.u_bc_dofs.index(d) for d in bc_dofs_x]
            aa_y = [self.u_bc_dofs.index(d) for d in bc_dofs_y]
            for i, j in enumerate(dofs_map_x):
                self.mapper[j] = [aa_x[i]]
            for i, j in enumerate(dofs_map_y):
                self.mapper[j].append(aa_y[i])
        elif self._dim == 3:
            dofs_x = v_bc.sub(0).dofmap().dofs()
            dofs_y = v_bc.sub(1).dofmap().dofs()
            dofs_z = v_bc.sub(2).dofmap().dofs()
            bc_dofs_x = [d for d in self.dofs if d in dofs_x]
            bc_dofs_y = [d for d in self.dofs if d in dofs_y]
            bc_dofs_z = [d for d in self.dofs if d in dofs_z]
            coords_bc_x = self.dofs_to_coords[bc_dofs_x]
            coords_bc_y = self.dofs_to_coords[bc_dofs_y]
            coords_bc_z = self.dofs_to_coords[bc_dofs_z]
            dofs_map_x = find_mixup(points, coords_bc_x, tol=tol)
            dofs_map_y = find_mixup(points, coords_bc_y, tol=tol)
            dofs_map_z = find_mixup(points, coords_bc_z, tol=tol)
            if not all(d!=-1 for d in dofs_map_x+dofs_map_y+dofs_map_z):
                ## in view of the implementation of the method "find_mixup"
                raise ValueError('At least one of the points does not match any node of the mesh corresponding to the specified function-space.')
            aa_x = [self.u_bc_dofs.index(d) for d in bc_dofs_x]
            aa_y = [self.u_bc_dofs.index(d) for d in bc_dofs_y]
            aa_z = [self.u_bc_dofs.index(d) for d in bc_dofs_z]
            for i, j in enumerate(dofs_map_x):
                self.mapper[j] = [aa_x[i]]
            for i, j in enumerate(dofs_map_y):
                self.mapper[j].append(aa_y[i])
            for i, j in enumerate(dofs_map_z):
                self.mapper[j].append(aa_z[i])
        
        if vals is not None:
            self.assign(vals)
            
    def assign(self, vals):
        """
        vals:
            List of np.array each being the value of BC at the corresponding point: each entry might be a scalar or an np.array depending on self._dim
        """
        if self._dim>1:
            assert(self._dim == len(vals[0])) # the given "vals" (of BCs) must be consistent with space_function
        self.u_bc.vector()[np.array(self.mapper).flatten()] = np.array(vals).flatten()
            # Shorter way of:
            # if self._dim == 1:
            #     for i, j in enumerate(self.mapper):
            #         self.u_bc.vector()[j] = vals[i] # bc_val is given as scalar
            # elif self._dim == 2:
            #     for i, j in enumerate(self.mapper):
            #         self.u_bc.vector()[j[0]] = vals[i][0]
            #         self.u_bc.vector()[j[1]] = vals[i][1]
            # elif self._dim == 3:
            #     for i, j in enumerate(self.mapper):
            #         self.u_bc.vector()[j[0]] = vals[i][0]
            #         self.u_bc.vector()[j[1]] = vals[i][1]
            #         self.u_bc.vector()[j[2]] = vals[i][2]
        u_bc_sub = self.u_bc.copy(deepcopy=True)
        for i in self.sub_ids:
            u_bc_sub = u_bc_sub.split()[i]
        self.bc.set_value(u_bc_sub)

class ValuesEvolverInTime:
    def __init__(self, assigner, values, checkpoints, _method='linear'):
        self.assigner = assigner # a callable
        self.values = values
            # A list of np.arrays, each being values of self.bc at particular DOFs (domain).
        self.checkpoints = checkpoints # values of time corresponding to columns of values
        self._method = _method # by default, a linear interpolation between time-steps is done.
        
        assert np.all(np.diff(np.array(checkpoints)) > 0) # checkpoints must be all increasing
        assert(len(checkpoints)==len(values))
    
    def __call__(self, t):
        if self._method=='linear':
            if (t <= self.checkpoints[0]):
                val_t = (t / self.checkpoints[0]) * np.array(self.values[0])
            else:
                for i in range(len(self.checkpoints)-1):
                    if (t>self.checkpoints[i] and t<=self.checkpoints[i+1]):
                        a = (t - self.checkpoints[i]) / (self.checkpoints[i+1] - self.checkpoints[i])
                        b = (self.checkpoints[i+1] - t) / (self.checkpoints[i+1] - self.checkpoints[i])
                        val_t = a * np.array(self.values[i+1]) + b * np.array(self.values[i])
                        break
            self.assigner(val_t)
    

class LocalProjector:
    def __init__(self, expr, V, dxm):
        """
        expr:
            expression to project
        V:
            quadrature function space
        dxm:
            dolfin.Measure("dx") that matches V
        """
        dv = df.TrialFunction(V)
        v_ = df.TestFunction(V)
        a_proj = df.inner(dv, v_) * dxm
        b_proj = df.inner(expr, v_) * dxm
        self.solver = df.LocalSolver(a_proj, b_proj)
        self.solver.factorize()

    def __call__(self, u):
        """
        u:
            function that is filled with the solution of the projection
        """
        self.solver.solve_local_rhs(u)


def compute_residual(F, bcs_dofs, reaction_dofs=[], logger=None, u_sol=None, u_F=None, write_residual_vector=False):
    """
    F:         Form of the variational problem (F==0)
    bcs_dofs:  The global DOFs of Dirichlet boundary conditions of the variational problem
    u_sol:     The solution at which the residual is computed.
                  It can be either of: FEniCS Function, list, np.ndarray (size must be compatible with u_F).
    u_F:       The Function object appearing in "F"
    
    NOTE:
        In default case (u_sol=None, u_F=None), the residual is computed as for the already computed Function appearing in "F".
    """
    if u_sol is not None: # we must assign u_sol to u_F object
        if u_F is None:
            print('ERROR: The Function object of the variational problem must be specified for the requested assignment.')
        else:
            if type(u_sol)==list or type(u_sol)==np.ndarray:
                f_space = u_F.function_space()
                u_sol = create_fenics_function(u_sol, f_space)
            u_F.assign(u_sol)
    res = list(df.assemble(F).get_local())
    reaction_forces_groups = []
    for r_dofs in reaction_dofs: # reaction_dofs is potentially a list of lists of DOFs
        reaction_forces_groups.append( [res[i] for i in r_dofs] )
    res = [r for i, r in enumerate(res) if i not in bcs_dofs]
    b_norm = np.linalg.norm(res)
    if logger is not None:
        logger.debug('\nComputed residual norm excluding all Dirichlet BCs. = ' + str(b_norm))
        if write_residual_vector:
            logger.debug('\nComputed residual vector excluding Dirichlet BCs.:\n' + str(res))
        logger.debug('\nComputed residual at given reaction DOFs (reaction forces) = ' + str(reaction_forces_groups))
    return reaction_forces_groups, b_norm

def create_fenics_function(f_vals, f_space):
    """
    Create a FEniCS Function object given f_vals in form of list or np.array
    """
    if type(f_vals)==list:
        f_vals = np.array(f_vals)
    f = df.Function(f_space)
    f.vector().set_local(f_vals)
    return f

def update_BCs_of_t(bcs, t): # So far, not used
    for bc in bcs:
        pass # to be developed (depending on whether bc.value() expression had any 't' parameter)

def boundary_condition(i, u, x):
    bc = df.DirichletBC(i, u, x)
    bc_dofs = [key for key in bc.get_boundary_values().keys()] # The global DOFs of bc
    return bc, bc_dofs

def boundary_condition_pointwise(i, u, x):
    ## IMPORTANT: Definition of "x" must not have any "on_boundary", otherwise pointwise BC would not be created.
    bc = df.DirichletBC(i, u, x, method='pointwise')
    bc_dofs = [key for key in bc.get_boundary_values().keys()] # The global DOFs of bc
    return bc, bc_dofs

def bc_bar_both_ends(i, u_left=df.Constant(0.), u_right=df.Constant(1.0), L_bar=1.0, tol=1e-14):
    def boundary_left(x):
        return df.near(x[0], 0., tol)
    def boundary_right(x):
        return df.near(x[0], L_bar, tol)
    bc_left, bc_left_dofs = boundary_condition(i, u_left, boundary_left)
    bc_right, bc_right_dofs = boundary_condition(i, u_right, boundary_right)
    
    bcs = [bc_left, bc_right]
    bcs_dofs = bc_left_dofs + bc_right_dofs
    
    return bcs, bcs_dofs

def write_to_xdmf(field, xdmf_name, xdmf_path=None):
    if xdmf_path is None:
        xdmf_path = './'
    make_path(xdmf_path)
    f = df.XDMFFile(xdmf_path + xdmf_name)
    f.write(field)
    
def marked_mesh_function_from_domains(domains_list, mesh, dim, tp='size_t', start_marker=1):
    mesh_function = df.MeshFunction(tp, mesh, dim)
    marker = start_marker
    markers = []
    for dom in domains_list:
        markers.append(marker)
        dom.mark(mesh_function, marker)
        marker += 1
    return mesh_function, markers

def measures_of_marked_mesh_function(mesh_function):
    mesh = mesh_function.mesh()
    dx = df.Measure('dx', domain = mesh, subdomain_data = mesh_function)
    ds = df.Measure('ds', domain = mesh, subdomain_data = mesh_function)
        
    ### alternative
    # dx = df.Measure('dx')[mesh_function]
    # ds = df.Measure('ds')[mesh_function]
    
    return dx, ds

def points_at(XY, _eps=1e-14):
    """
    Return a dolfin.SubDomain object representing all the points listed in XY (each being an np.array)
    """
    _dim = len(XY[0])
    def _inside(x):
        b = False
        if _dim == 1:
            for p in XY:
                b = b or df.near(x[0], p[0], _eps)
                if b:
                    break
        elif _dim == 2:
            for p in XY:
                b = b or (df.near(x[0], p[0], _eps) and df.near(x[1], p[1], _eps))
                if b:
                    break
        elif _dim == 3:
            for p in XY:
                b = b or (df.near(x[0], p[0], _eps) and df.near(x[1], p[1], _eps) and df.near(x[2], p[2], _eps))
                if b:
                    break
        return b
    
    return df.AutoSubDomain(_inside)

def unique_points_at_dofs(V, dofs, points_tol=None):
    if points_tol is None:
        points_tol = V.mesh().rmin() / 1000.
    all_ps = V.tabulate_dof_coordinates()
    ps = all_ps[dofs, :]
    return add_unique_points(new_points=ps, points0=[], _eps=points_tol)[0]

def dofs_at(points, V, i, tol=None):
    """
    IMPORTANT:
        "V" is a total function-space containing all DOFs (cannot be a sub-space)
        "i" must be a "subspace" which has at-most ONE DOF at any point
    """
    if tol is None:
        tol = V.mesh().rmin() / 1000.
    i_dofs = i.dofmap().dofs()
    V_coords = V.tabulate_dof_coordinates()
    i_coords = V_coords[i_dofs]
    dofs_ids = find_among_points(points=points, points0=i_coords, tol=tol)
    if any([_id is None for _id in dofs_ids]):
        raise ValueError(f"At least one point does not coincide with any node.")
    dofs = [i_dofs[_id] for _id in dofs_ids]
    return dofs
