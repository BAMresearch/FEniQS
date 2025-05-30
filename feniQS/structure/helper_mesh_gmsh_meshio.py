import numpy as np
import gmsh
import os

from feniQS.general.general import make_path, CollectPaths, find_among_points

pth_helper_mesh_gmsh_meshio = CollectPaths('./feniQS/structure/helper_mesh_gmsh_meshio.py')

"""
f: file (only file name + format)
ff: full file (path + file name + format)
"""

def scale_mesh_meshio(mesh_or_mesh_file, scale: float \
                    , ff_scaled_mesh: str = None, binary : bool = True \
                    , centralize: bool = False):
    import meshio
    cs0, cells0 = get_mesh_points_and_cells(mesh_or_mesh_file=mesh_or_mesh_file)
    if centralize:
        for i in range(3):
            c = 0.5 * (max(cs0[:,i]) + min(cs0[:,i]))
            cs0[:,i] -= c
    cs = scale * cs0
    if ff_scaled_mesh is not None:
        _dir = os.path.dirname(ff_scaled_mesh)
        make_path(_dir)
        meshio.write_points_cells(filename=ff_scaled_mesh \
                                , points=cs, cells=cells0 \
                                , binary=binary)
    return meshio.Mesh(points=cs, cells=cells0)

def get_surface_mesh_from_volume_mesh(mesh_or_mesh_file, ff_mesh_surf=None):
    """
    Creates/returns a surface mesh which contains ONLY and ALL triangles which by themselves belong to
        ONLY ONE tetrahedral element.
    If ff_mesh_surf is not None, the resultant mesh is stored in such file.
    """
    import meshio
    cs0, cells0 = get_mesh_points_and_cells(mesh_or_mesh_file=mesh_or_mesh_file \
                                            , meshio_cell_type='tetra')
    triangles = dict()
    for c in cells0:
        for t in [[c[0], c[1], c[2]],
                  [c[0], c[1], c[3]],
                  [c[0], c[2], c[3]],
                  [c[1], c[2], c[3]]]:
            t = tuple(sorted(t))
            try:
                del triangles[t] # a duplicate triangle must NOT belong to surface mesh.
            except KeyError:
                triangles[t] = list(t)
    triangles = [t for t in triangles.values()]
    cs, cells, _ = remove_isolated_nodes(cs=cs0, cells=triangles)
    if ff_mesh_surf is not None:
        meshio.write_points_cells(filename=ff_mesh_surf, points=cs, cells={'triangle': cells})
    return meshio.Mesh(points=cs, cells={'triangle': cells})

def remove_isolated_nodes(cs, cells):
    """
    cs: mesh coordinates (np.array) of a certain mesh
    cells: cell connectivities (np.array) of the same mesh (consistent with cs)
    This method removes potential nodes (among 'cs') that do not belong to any cell,
    and returns unique nodes other than such nodes, the updated cell connectivities, plus
    the IDs of those unique (non-isolated) nodes at the original 'cs'.
    """
    import numpy as np
    node_IDs_original = [] # in all the cells
    for c in cells:
        node_IDs_original += list(c)
    unique_cs_IDs_original = list(set(node_IDs_original))
    new_node_IDs_from_original_IDs = {int(_id): i for (i, _id) in enumerate(unique_cs_IDs_original)}
    unique_cs = cs[unique_cs_IDs_original,:]
    updated_cells = np.array([[new_node_IDs_from_original_IDs[ci] for ci in c] for c in cells])
    return unique_cs, updated_cells, unique_cs_IDs_original

def remove_missing_cells(cells, subset_cs_ids):
    """
    cells: cell connectivities (np.array) of a certain mesh.
    subset_cs_ids: A subset of IDs among all the node IDs referred in cells
        (the latter ranges from 0 to N).
    This method removes cells with at-least one connectivity that is missing in subset_cs_ids,
    and returns the updated cell connectivities.
    """
    subset_ids = {j: i for i, j in enumerate(subset_cs_ids)}
    updated_cells = []
    for c in cells:
        try:
            updated_cells.append([subset_ids[n] for n in c])
        except KeyError:
            pass
    return np.array(updated_cells)

def extract_a_sub_mesh(mesh0_or_mesh0_file, meshio_cell_type \
                     , sub_mesh_cs_ids=None, callable_is_a_point_in_sub_mesh=None \
                     , ff_sub_mesh: str = None, mesh0_node_groups: dict = None):
    cs0, cells0 = get_mesh_points_and_cells(mesh_or_mesh_file=mesh0_or_mesh0_file \
                                            , meshio_cell_type=meshio_cell_type)
    if sub_mesh_cs_ids is None:
        if callable_is_a_point_in_sub_mesh is None:
            raise ValueError("At least one of 'sub_mesh_cs_ids' and 'callable_is_a_point_in_sub_mesh' must be not None.")
        else:
            assert callable(callable_is_a_point_in_sub_mesh)
            sub_mesh_cs_ids = [i for i, c in enumerate(cs0) if callable_is_a_point_in_sub_mesh(c)]
    else:
        assert isinstance(sub_mesh_cs_ids, list)
    sub_mesh_cells = remove_missing_cells(cells=cells0, subset_cs_ids=sub_mesh_cs_ids)
    sub_mesh_cs = cs0[sub_mesh_cs_ids, :]
    sub_mesh_cs_unique, sub_mesh_cells_unique, unique_cs_IDs_original = remove_isolated_nodes(
        cs=sub_mesh_cs, cells=sub_mesh_cells)
    sub_mesh_unique_cs_IDs_at_mesh0 = [sub_mesh_cs_ids[_i] for _i in unique_cs_IDs_original]
    _new_node_IDs_from_original_IDs = {int(_id): i for (i, _id) in enumerate(sub_mesh_unique_cs_IDs_at_mesh0)}
    if mesh0_node_groups is None:
        sub_mesh_node_groups = None
    else:
        sub_mesh_node_groups = dict() # The node IDs will be based on the nodes of sub_mesh itself.
        for k, v in mesh0_node_groups.items():
            sub_mesh_node_groups[k] = []
            for vi in v:
                try:
                    sub_mesh_node_groups[k].append(_new_node_IDs_from_original_IDs[vi])
                except KeyError:
                    sub_mesh_node_groups[k].append(None) # To keep a one-to-one correspondance
    import meshio
    if ff_sub_mesh is not None:
        _dir = os.path.dirname(ff_sub_mesh)
        make_path(_dir)
        meshio.write_points_cells(filename=ff_sub_mesh, points=sub_mesh_cs_unique \
                                  , cells={meshio_cell_type: sub_mesh_cells_unique})
    return meshio.Mesh(points=sub_mesh_cs_unique, cells={meshio_cell_type: sub_mesh_cells_unique}) \
        , sub_mesh_unique_cs_IDs_at_mesh0, sub_mesh_node_groups

def get_fenics_mesh_object(mesh_or_mesh_file):
    import dolfin as df
    import meshio
    def _get_dolfin_mesh_xdmf(file_xdmf):
        _mesh = df.Mesh()
        with df.XDMFFile(file_xdmf) as ff:
            ff.read(_mesh)
        return _mesh
    def _get_dolfin_mesh_meshio(meshio_mesh):
        import os
        _tmp_file = './tmp_mesh.xdmf'
        meshio.write(_tmp_file, meshio_mesh)
        _mesh = _get_dolfin_mesh_xdmf(_tmp_file)
        os.remove(_tmp_file)
        return _mesh
    if isinstance(mesh_or_mesh_file, str):
        if mesh_or_mesh_file.endswith('.xdmf'):
            mesh = _get_dolfin_mesh_xdmf(mesh_or_mesh_file)
        else:
            _m = meshio.read(mesh_or_mesh_file)
            mesh = _get_dolfin_mesh_meshio(_m)
    elif isinstance(mesh_or_mesh_file, meshio.Mesh):
        mesh = _get_dolfin_mesh_meshio(mesh_or_mesh_file)
    elif isinstance(mesh_or_mesh_file, df.Mesh):
        mesh = mesh_or_mesh_file
    else:
        raise ValueError(f"The input mesh_or_mesh_file='{mesh_or_mesh_file}' is neither a mesh file nor a recognized mesh object.")
    return mesh

def get_mesh_statistics(mesh_or_mesh_file):
    mesh = get_fenics_mesh_object(mesh_or_mesh_file=mesh_or_mesh_file)
    from feniQS.fenics_helpers.fenics_functions import get_element_volumes
    import numpy as np
    element_volumes = get_element_volumes(mesh)
    return {
        'volume_full': float(np.sum(element_volumes)),
        'volume_cell_mean': float(np.mean(element_volumes)),
        'volume_cell_min': float(np.min(element_volumes)),
        'volume_cell_max': float(np.max(element_volumes)),
        'num_nodes': mesh.num_vertices(),
        'num_cells': mesh.num_cells(),
        'num_edges': mesh.num_edges(),
        'num_faces': mesh.num_faces(),
        'num_facets': mesh.num_facets(),
        'r_min': mesh.rmin(),
        'r_max': mesh.rmax(),
    }

def get_mesh_volume(mesh_or_mesh_file, sum_over_cells=True):
    """
    mesh_or_mesh_file:
        either a mesh file (supported by meshio) or a mesh object (of dolfin or meshio).
        It is preferred to input either a FEniCS mesh object, or an xdmf mesh file.
    For 2D (planner) elements of a mesh, the method computes the total area of those elements.
    """
    mesh = get_fenics_mesh_object(mesh_or_mesh_file=mesh_or_mesh_file)
    if sum_over_cells:
        from feniQS.fenics_helpers.fenics_functions import get_element_volumes
        return sum(get_element_volumes(mesh))
    else:
        import dolfin as df
        return df.assemble(1. * df.dx(mesh))

def get_meshQualityMetrics_triangular(mesh_file=None, points=None, cells=None \
                                    , _path_yaml=None, _name_yaml='mesh_quality' \
                                    , _path_plots=None, _name_plots='mesh_quality', _show_plot=True):
    """
    Inputs must contain either the 'mesh_file' (supported by meshio) or both of 'points' and 'cells'.
    The mesh (or points/cells) must contain triangle elements.
    _path_plots: If not None, the plots of mesh quality metrics will be stored.
    _path_yaml: If not None, the mesh quality metrics will be stored in yaml files.
    """
    if mesh_file is None:
        if (points is None) or (cells is None):
            raise ValueError(f"Specify either the mesh_file or both of points and cells.")
        if points.shape[1]==2:
            points = np.concatenate((points, np.zeros((points.shape[0], 1))), 1)
        if cells.shape[1]!=0:
            raise ValueError(f"The input cells do not represent triangles.")
    else:
        if points is not None:
            print(f"WARNING (mesh quality metrics):\n\tThe input points were ignored (points were taken from mesh_file).")
        if cells is not None:
            print(f"WARNING (mesh quality metrics):\n\tThe input cells were ignored (cells were taken from mesh_file).")
        import meshio
        _mesh = meshio.read(mesh_file)
        points = _mesh.points
        try:
            cells = _mesh.cells_dict['triangle']
        except KeyError:
            raise ValueError(f"The input mesh_file has no triangle elements.")
    tol = max(max(points[:,0]) - min(points[:,0])
              , max(points[:,1]) - min(points[:,1])
              , max(points[:,2]) - min(points[:,2])) / points.shape[0] / 1000.
    qualities = {'aspect_ratio': [], 'min_angle': [], 'area': []}
    for cc in cells:
        p1, p2, p3 = points[cc[0]], points[cc[1]], points[cc[2]]
        # Compute edge lengths
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        # Avoid duplicate nodes
        if min(a, b, c) < tol:
            raise ValueError(f"Triangle {cc} has duplicate/coinciding nodes!")
        # Aspect Ratio: (max edge length) / (min edge length)
        aspect_ratio = max(a, b, c) / min(a, b, c)
        # Compute angles
        cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_beta  = (a**2 + c**2 - b**2) / (2 * a * c)
        cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
        # Avoid precision issues with acos
        angles = np.arccos(np.clip([cos_alpha, cos_beta, cos_gamma], -1.0, 1.0))
        min_angle = np.degrees(np.min(angles))
        # Compute area using Heron's formula
        s = 0.5 * (a + b + c)
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        qualities['aspect_ratio'].append(aspect_ratio)
        qualities['min_angle'].append(min_angle)
        qualities['area'].append(area)
    if _path_yaml is not None:
        import os, yaml
        from feniQS.general.yaml_functions import yamlDump_array
        if not os.path.exists(_path_yaml):
            os.mkdir(_path_yaml)
        for k, q in qualities.items():
            yamlDump_array(np.array(q), f"{_path_yaml}{_name_yaml}_{k}.yaml")
    if _path_plots is not None:
        import os
        import matplotlib.pyplot as plt
        if not os.path.exists(_path_plots):
            os.mkdir(_path_plots)
        for k in ['aspect_ratio', 'min_angle', 'area']:
            aa = k.replace('_', ' ')
            plt.figure()
            plt.plot(qualities[k], linestyle='', marker='.')
            plt.xlabel('Cell ID')
            plt.ylabel(aa.capitalize())
            plt.title(f"Mesh quality metric ({aa})")
            plt.savefig(f"{_path_plots}{_name_plots}_{k}.png", bbox_inches='tight', dpi=500)
            if _show_plot:
                plt.show()
            plt.close()
    return qualities

def get_mesh_points_and_cells(mesh_or_mesh_file, meshio_cell_type=None):
    """
    mesh_or_mesh_file:
        either a mesh file (supported by meshio) or a mesh object (of dolfin or meshio).
    Returns:
        mesh 'nodes' as a numpy array object.
        mesh 'cells' either as a dictionary (if meshio_cell_type is None),
            or as a numpy array object (for the meshio_cell_type specified).
    The meshio_cell_type (if specified) must be according to the meshio library.
    """
    import meshio
    if isinstance(mesh_or_mesh_file, str):
        mesh_or_mesh_file = meshio.read(mesh_or_mesh_file)
    if isinstance(mesh_or_mesh_file, meshio.Mesh):
        cs = mesh_or_mesh_file.points
        cells = mesh_or_mesh_file.cells_dict
        if meshio_cell_type is not None:
            cells = cells[meshio_cell_type]
    else:
        import dolfin as df
        if isinstance(mesh_or_mesh_file, df.Mesh):
            cs = mesh_or_mesh_file.coordinates()
            cells = mesh_or_mesh_file.cells()
        else:
            raise ValueError(f"The input mesh_or_mesh_file='{mesh_or_mesh_file}' is neither a mesh file nor a recognized mesh object.")
    return cs, cells

def _get_extended_mesh_data(points0, cells0 \
                            , points_to_extend, cells_to_extend \
                            , l, tol, direction='x'):
    """
    Crucial:
        The node IDs referred to in "cells_to_extend" must be associated with the rows of 'points_to_extend'.
    """
    assert direction in ['x', 'y', 'z']
    directions_IDs = {'x':0, 'y':1, 'z':2}
    points_extended0 = points_to_extend.copy()
    points_extended0[:, directions_IDs[direction]] += l
    ids = find_among_points(points=points_extended0, points0=points0, tol=tol)
    points_extended = points0.copy()
    id_last_p = points_extended.shape[0] - 1
    points_extended_IDs = dict()
    for i, id in enumerate(ids):
        if id is None: # a new extended point
            points_extended = np.vstack((points_extended, points_extended0[i, :]))
            points_extended_IDs[i] = id_last_p + 1
            id_last_p += 1
        else:
            points_extended_IDs[i] = int(id)
    cells_extended = np.empty_like(cells_to_extend)
    for i,c in enumerate(cells_to_extend):
        cells_extended[i,:] = [points_extended_IDs[j] for j in c]
    cells_extended = np.concatenate((cells0, cells_extended), axis=0)
    return points_extended, cells_extended

def extend_mesh_periodically_meshio(mesh0_or_mesh0_file, mesh_file \
                                    , meshio_cell_type, n, tol=None \
                                    , translation=None):
    """
    IMPORTANT:
        Thie method does NOT care if the original mesh (mesh0_or_mesh0_file) is periodic!
    
    mesh0_or_mesh0_file: corresponds to the original mesh,
        and is either a mesh file (supported by meshio) or a mesh object (of dolfin or meshio).
    mesh_file: the full file name for the extended mesh to be stored (vie meshio).
    tol: the tolerance for detecting duplicate nodes.
    meshio_cell_type: the type of cells of the original mesh (according to meshio library)
        For now it is working when there is only one meshio_cell_type.
    n:
        if integer: that many copies of the input mesh at each spatial direction.
            NOTE: n=1 implies no extension (1 copy) and returns the original mesh.
        if tuple/list: respectively in directions 'x', 'y' and 'z'.
        if dictionary:
            keys are 'x' and/or 'y' and/or 'z',
            values are integers, indication the number of extensions at respective direction
    translation:
        if not None, must be a tuple/list/array of length 3, describing the translation
        of the final extended mesh in three x,y,z directions.
    """
    cs, cells = get_mesh_points_and_cells(mesh_or_mesh_file=mesh0_or_mesh0_file
                                          , meshio_cell_type=meshio_cell_type)
    c_dim = cs.shape[1]
    directions_IDs = {'x':0, 'y':1, 'z':2}
    cs_RVE0 = cs + 0. # deepcopy
    RVE_node_IDs = {'000': list(range(cs_RVE0.shape[0]))}
    
    ls = dict()
    for k, v in directions_IDs.items():
        if v < c_dim:
            ls[k] = float(np.max(cs[:,v]) - np.min(cs[:,v]))
    
    if isinstance(n, int):
        ns = {k: n for k,v in directions_IDs.items() if v<c_dim}
    elif isinstance(n, list) or isinstance(n, tuple):
        if len(n)>c_dim:
            raise ValueError(f"The input n={n} is ambiguous, since its length exceeds the dimension of input mesh nodes (={c_dim}).")
        ns = dict()
        for i, v in enumerate(n):
            ns[['x', 'y', 'z'][i]] = v
    else:
        assert isinstance(n, dict)
        assert all([k in ['x', 'y', 'z'] for k in n.keys()])
        if len(n)>c_dim:
            raise ValueError(f"The input n={n} is ambiguous, since its items do not match with the dimension of input mesh nodes (={c_dim}).")
        ns = n
    assert all([isinstance(v, int) for v in ns.values()])
    
    if tol is None:
        tol = 0.
        for l in ls.values():
            tol = max(tol, l)
        tol = tol / cs.shape[0] / 1000.
    
    for k, v in ns.items():
        cs0 = cs.copy()
        cells0 = cells.copy()
        l = ls[k]
        for i in range(1, v):
            cs, cells = _get_extended_mesh_data(points0=cs, cells0=cells \
                                                , points_to_extend=cs0, cells_to_extend=cells0 \
                                                , l=i*l, tol=tol, direction=k)
    
    ## The following part extracts the node IDs per each extended RVE,
    # and is for now working only when the extension is in all 3 directions (for a 3D RVE).
    RVE_i = 0
    RVE_nodes_labels = np.zeros(cs.shape[0], dtype=int)
    RVE_nodes_labels[RVE_node_IDs['000']] = RVE_i # The labels have to be numbers (string is not possible).
    for i_x in range(ns['x']):
        x1 = np.min(cs[:,0]) + i_x * ls['x']
        x2 = np.min(cs[:,0]) + (i_x + 1) * ls['x']
        _where_x = np.logical_and(x1-tol<cs[:,0], cs[:,0]<x2+tol)
        for i_y in range(ns['y']):
            y1 = np.min(cs[:,1]) + i_y * ls['y']
            y2 = np.min(cs[:,1]) + (i_y + 1) * ls['y']
            _where_y = np.logical_and(y1-tol<cs[:,1], cs[:,1]<y2+tol)
            _where_xy = np.logical_and(_where_x, _where_y)
            for i_z in range(ns['z']):
                if not (i_x==0 and i_y==0 and i_z==0):
                    RVE_i += 1
                    z1 = np.min(cs[:,2]) + i_z * ls['z']
                    z2 = np.min(cs[:,2]) + (i_z + 1) * ls['z']
                    _where_z = np.logical_and(z1-tol<cs[:,2], cs[:,2]<z2+tol)
                    _where = np.logical_and(_where_xy, _where_z)
                    _where_IDs = np.where(_where)[0]
                    _cs_RVE = cs[_where, :]
                    _cs_RVE[:, 0] -= i_x * ls['x']
                    _cs_RVE[:, 1] -= i_y * ls['y']
                    _cs_RVE[:, 2] -= i_z * ls['z']
                    _ids = find_among_points(points=cs_RVE0, points0=_cs_RVE, tol=tol)
                    rve_label = f"{i_x}{i_y}{i_z}"
                    RVE_node_IDs[rve_label] = [int(_where_IDs[_i]) for _i in _ids]
                    RVE_nodes_labels[RVE_node_IDs[rve_label]] = int(RVE_i * (-1) ** (sum(int(d) for d in rve_label)))
                        # The labels have to be numbers (string is not possible).
    point_data = {'RVE_nodes': RVE_nodes_labels}

    if translation is not None:
        if len(translation)!=3:
            raise ValueError(f"Specify the translation of the mesh as a tuple, list, or array with a length of 3.")
        for i in range(3):
            cs[:, i] += translation[i]
    import meshio
    meshio.write_points_cells(mesh_file, cs, {meshio_cell_type: cells}
                              , point_data=point_data)
    mesh_meshio = meshio.Mesh(points=cs, cells={meshio_cell_type: cells}
                              , point_data=point_data)
    return mesh_meshio, RVE_node_IDs

def extrude_triangled_mesh_by_meshio(ff_mesh_surface, ff_mesh_extruded, dz, res=1 \
                                     , node_collections={}, tol=None):
    """
    ff_mesh_surface:
        A file (full file name) that stores a surface mesh with triangle elements.
        This mesh can also be over a 3-D surface (not necessarily on a plane), but
        it must contain ONLY triangle cells.
    
    ff_mesh_extruded:
        The target file (full file name) for storing the resultant 3-D mesh with tetrahedral elements.
        
    dz:
        Either of the following:
        - A float number, indicating the extrusion in z-direction, applied to all nodes of the surface mesh,
        - A tuple/list of two numbers (usually positive/negative), indicating the extrusion in two opposite
            directions along the z-axis, applied to all nodes of the surface mesh,
        - A callable with the input argument being coordinates of a point on the surface mesh, which then
            returns either of the following:
            - one (out-of-plan) vector,
            - a tuple/list of two (out-of-plan) vectors.
            Each of such out-of-plane vectors potentially pertains to the extrusion of the triangled
            surface mesh on one side of it.
        NOTEs:
        - The "z" direction and the vectors mentioned above are considered in the global coordinate system.
        - If "dz" is a callable, it must return the same number of vectors (either one or two)
            for all nodes of the triangled mesh. This means, this python method cannot selectively
            extrude a selected part of the surface mesh at one/two sides, differently than other parts.
            Also note that, this requirement is not checked/asserted in this implementation, thus,
            a "dz" callable that does not meet this condition may create an unexpected mesh, potentially
            without raising any errors/warnings.
    
    res:
        Resolution (number of layers) per each side of extrusion, which is either one integer (for
        one-side extrusion) or a tuple/list of two integers (for double-side extrusion).
        This number of extrusion-sides must be consistent with the input "dz" (see above).
    
    node_collections:
        A dictionary whose each value is a collection of nodes (np.ndarray of coordinates)
            , such that all of those coordinates belong to the nodes of the mesh "ff_mesh_surface".
        During the extrusion, those collections are also extruded to their respective (extruded) collections.
        So, the method returns a so-called extruded_node_collections.

    tol:
        A tolerance used only for finding node_collections among the surface mesh's nodes.
    
    Reference:
    This python method is based on the procedure of "Tetrahedral decomposition of triangular prism".
    Specifically, see Fig.4 of http://dx.doi.org/10.1145/1186822.1073239 (also attached here as picture).
    """
    import meshio
    mesh = meshio.read(ff_mesh_surface)
    if any([c.type!='triangle' for c in mesh.cells]):
        raise ValueError(f"The input mesh must only contain triangle elements.")
    ps = mesh.points
    n_ps = ps.shape[0]
    if len(node_collections)>1 and (tol is None): # we set a sensible tolerance
        l = 0.
        for i in range(3):
            try:
                l += (max(ps[:, i]) - min(ps[:, i])) ** 2
            except IndexError:
                pass
        l = np.sqrt(l) # estimated diagonal of the bounding box containing the entire mesh
        tol = l / ps.shape[0] / 1000.
    extruded_node_collections_IDs = {}
    for k, v in node_collections.items():
        _ids = find_among_points(points=v, points0=ps, tol=tol)
        if any ([_id is None for _id in _ids]):
            raise ValueError(f"At least one node among the given node_collections does not belong to the surface mesh.")
        extruded_node_collections_IDs[k] = _ids
    if ps.shape[1]==2:
        ps3 = np.append(ps, np.zeros((n_ps,1)), axis=1)
    else:
        ps3 = ps
    
    if callable(dz):
        def dz_of_i(i):
            dz_i = dz(ps[i,:]) # a vector that reflects both direction and magnitude of extrusion
            if not isinstance(dz_i, (tuple, list,)):
                dz_i = (dz_i,)
            return dz_i
    else:
        def dz_of_i(i):
            return dz if isinstance(dz, (tuple, list,)) else (dz,)
    dz0 = dz_of_i(0)
    num_sides = len(dz0)
    if isinstance(res, int):
        res = [res] * num_sides
    assert len(res)==num_sides
    shift_nodes_side = [0, n_ps * res[0]]
    res_total = sum(res)
    ps_3d = np.concatenate((ps3, np.zeros((res_total*n_ps,3))), axis=0)
    
    dzs = [dz_of_i(i) for i in range(n_ps)]
    if not all([len(dz_i)==num_sides for dz_i in dzs]):
        raise ValueError("Extrusion must be applied to one or double sides for all nodes.")
    
    extruded_ids = dict()
        # The keys of 'extruded_ids' refer to the node IDs of the input 'ff_mesh_surface'
        # , which also are exactly the node IDs of the first 'n_ps' nodes in 'ps_3d'.
    for i, dz_i in enumerate(dzs):
        extruded_ids[i] = []
        for i_side, dz_side in enumerate(dz_i):
            for iz in range(1, res[i_side]+1):
                shift = shift_nodes_side[i_side] + iz * n_ps
                if isinstance(dz_side, (int, float,)):
                    ps_3d[i+shift, :2] = ps3[i,:2]
                    ps_3d[i+shift, 2] = iz*dz_side/res[i_side]
                else:
                    ps_3d[i+shift,:] = ps3[i,:] + iz*dz_side/res[i_side]
                extruded_ids[i].append(i+shift)
        for k in extruded_node_collections_IDs.keys():
            if i in extruded_node_collections_IDs[k]:
                extruded_node_collections_IDs[k] += extruded_ids[i]
    mesh.points = ps_3d
    mesh.cells[0].type = 'tetra'
    mesh.cells[0].dim = 3
    extruded_node_collections = {k: ps_3d[v,:] for k, v in extruded_node_collections_IDs.items()}
    
    mesh_2d_cell_data = mesh.cells[0].data
    n_cells_2d = mesh_2d_cell_data.shape[0]
    extruded_cells_ids = {i: [] for i in range(n_cells_2d)}
    mesh.cells[0].data = np.empty((res_total*n_cells_2d*3, 4), dtype=np.int32)
    shift_cells_side = [0, res[0]*n_cells_2d*3]
    for i_side in range(num_sides):
        for iz in range(0, res[i_side]):
            bb = 0 if iz==0 else shift_nodes_side[i_side]
            cc = shift_nodes_side[i_side] if iz==0 else 0
            mesh_2d_base = bb + mesh_2d_cell_data + n_ps * iz
            for i in range(n_cells_2d):
                cs_2d = mesh_2d_base[i]
                aa = np.sort(cs_2d) # sorted (very crucial)
                i0 = shift_cells_side[i_side] + (n_cells_2d*iz+i)*3
                mesh.cells[0].data[i0, :] = [aa[0], aa[1], aa[2], aa[2]+n_ps+cc]
                mesh.cells[0].data[i0+1, :] = [aa[0], aa[1], aa[2]+n_ps+cc, aa[1]+n_ps+cc]
                mesh.cells[0].data[i0+2, :] = [aa[0], aa[0]+n_ps+cc, aa[1]+n_ps+cc, aa[2]+n_ps+cc]
                extruded_cells_ids[i].extend([i0, i0+1, i0+2])
    
    meshio.write(ff_mesh_extruded, mesh)
    return extruded_node_collections, extruded_ids, extruded_cells_ids

def get_xdmf_mesh_by_meshio(ff_mesh, geo_dim, path_xdmf=None, _msg=True):
    """
    ff_mesh:
        the 'full' name of the file containing an input mesh (with any format that meshio supports).
    path_xdmf:
        the path in which the file of the generated/converted xdmf-mesh is stored. If None, the same path of ff_mesh is used.
    NOTEs:
        - The name of the converted file is the same as the input file's.
        - The 'geo_dim' is required to remove unnecessary nodal coordinates (possibly the 'y' and/or 'z' coordinates).
    """
    import meshio
    fn, fe = os.path.splitext(os.path.basename(ff_mesh))
    if fe == '.xdmf':
        if _msg:
            print(f"The given mesh is already in '.xdmf' format.")
        file_mesh_xdmf = ff_mesh
    else:
        if _msg:
            print(f"The given mesh is converted to '.xdmf' format.")
        if path_xdmf is None:
            path_xdmf = os.path.dirname(ff_mesh) + '/'
        else:
            if not os.path.exists(path_xdmf):
                os.makedirs(path_xdmf)
        file_mesh_xdmf = path_xdmf + fn + '.xdmf'
        ## Convert the mesh
        meshio_mesh_ = meshio.read(ff_mesh) # Has allways points in x-y-z coordinates.
        if meshio_mesh_.points.shape[1]==geo_dim:
            meshio_mesh = meshio_mesh_
        else:
            meshio_mesh = meshio.Mesh(points=meshio_mesh_.points[:, :geo_dim], cells=meshio_mesh_.cells) # Remove unnecessary coordinates.
        cell_type = {1: 'line', 2: 'triangle', 3: 'tetra'}[geo_dim]
        meshio.write_points_cells(file_mesh_xdmf, meshio_mesh.points \
                                , cells={cell_type: meshio_mesh.get_cells_type(cell_type)})
        # The following two would fail to be read back in FEniCS for case of tetra cell type!
    # meshio.write_points_cells(ff_xdmf, meshio_mesh.points, meshio_mesh_.cells)
    # meshio.write(ff_xdmf, meshio_mesh)
    return file_mesh_xdmf

def geometric_resolution_over_length(l, scale, l0):
    """
    l : float
        total length of a line
    scale : float
        total scale of mesh size; i.e. from first element size (l0) to last element size (ln)
    l0 : float
        size of the first elemen
    Returns
    -------
    n : int
        number of elements (segments) over the line
    r : float
        ratio between sizes of two successive elements
    l0_corrected : float
        corrected value of l0 (l0 cannot be perfectly achieved if we force 'scale' between first and last element sizes)
    """
    if abs(abs(scale) - 1.) < 1e-3:
        n = int(round(l / l0))
        r = 1.
        l0_corrected = l0
    else:
        b = l / l0
        if b <= 1.:
            n = 1; r = scale; l0_corrected = l0
        else:
            r = (b - 1.) / (b - scale)
            n = int(1 + round(np.log(scale) / np.log(r)))
            if n > 1:
                r = scale ** (1. / (n - 1)) # corrected
            else:
                r = scale
            l0_corrected = l * (r - 1.) / (r**n - 1.)
    return n, r, l0_corrected

def gmshAPI_generate_bcc_lattice(r_strut      = 0.5,
                                 l_rve        = 3.333333,
                                 n_rve        = 2,
                                 l_cell       = 0.2,
                                 add_plates   = True,
                                 shape_name   = "bcc",
                                 _path        = './',
                                 _name        = 'parametric_bcc',
                                ):
    """
    This function creates a solid mesh of a body-centered-cubic lattice
    structure (bcc). It can either be a multi-layer-lattice (mll) with
    n_rve > 1 or a single representative volule element (rve) with n_rve = 1.
    """
    
    ### PREPARE GMSH ##########################################################
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", l_cell);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", l_cell);
    gmsh.model.add("BCC")
    gmsh.model.setCurrent("BCC")
    geo = gmsh.model.occ
    
    ### CREATE SINGLE RVE OF BCC ##############################################
    geo.addCylinder(0.0,   0.0,    0.0,   l_rve,   l_rve,  l_rve,  r_strut, 1)
    geo.addCylinder(l_rve, 0.0,    0.0,   -l_rve,  l_rve,  l_rve,  r_strut, 2)
    geo.addCylinder(0.0,   l_rve,  0.0,   l_rve,   -l_rve, l_rve,  r_strut, 3)
    geo.addCylinder(0.0,   0.0,    l_rve, l_rve,   l_rve,  -l_rve, r_strut, 4)
    geo.fuse([(3, 2), (3, 3), (3, 4)], [(3, 1)], 5)
    geo.addBox(0.0, 0.0, 0.0, l_rve, l_rve, l_rve, 6)
    geo.intersect([(3, 6)], [(3, 5)], 7)
    c = 7
    
    ### GENERATE COPIES FOR MULTI-LAYER BCC ###################################
    if n_rve > 1:
        translation = [[l_rve, 0.0, 0.0], [0.0, l_rve, 0.0], [0.0, 0.0, l_rve]]
        for i in range(3):
            dx, dy, dz = translation[i]
            for j in range(n_rve - 1):
                outDimTag = geo.copy([(3, c)]) #8
                geo.translate(outDimTag, dx, dy, dz)
                geo.fuse([(3, c + 1)], [(3, c)], c + 2)
                c += 2
                print(n_rve, 'copies in dimension', i, 'were made.')
    ### ADD PLATES ############################################################
    if add_plates == True:
        l_mll = n_rve * l_rve
        h_plate = 0.4
        geo.addBox(0.0,    0.0,    0.0,    l_mll,   l_mll,  -h_plate,  100)
        geo.addBox(0.0,    0.0,    l_mll,  l_mll,   l_mll,  h_plate,   101)
        geo.fuse([(3, c)], [(3, 100), (3, 101)])
    
    ### CREATE MESH AND WRITE FILEs ########################################
    ff_msh = _path + '/' + _name + ".msh"
    geo.synchronize()
    mesh = gmsh.model.mesh.generate(3)
    gmsh.write(ff_msh)
    gmsh.finalize()
    
    return ff_msh

def gmshAPI_Lshape2D_mesh(lx, ly, wx, wy, res_x, res_y, embedded_nodes \
                          , el_size_min=None, el_size_max=None \
                        , _path='./', _name='lshape2D', write_geo=False):
    ## PARAMETERs
    geo_dim = 2

    ## FILEs
    make_path(_path)
    ff_geo = _path + '/' + _name + ".geo_unrolled" # if requested, writen after API.
    ff_msh = _path + '/' + _name + ".msh"

    ############### Gmsh API #################
    gmsh.initialize()
    model_tag = gmsh.model.add(_name)  # give the model a name
    ps_tags = []
    ls_tags = []
    curve_tags = []
    plane_tags = []

    _this = gmsh.model.occ
    # _this = gmsh.model.geo # Not sure, what is the difference to the above one!

    ps_tags.append(_this.addPoint(0., 0., 0.))
    ps_tags.append(_this.addPoint(0., ly, 0.))
    ps_tags.append(_this.addPoint(lx, ly, 0.))
    ps_tags.append(_this.addPoint(lx, ly-wy, 0.))
    ps_tags.append(_this.addPoint(wx, ly-wy, 0.))
    ps_tags.append(_this.addPoint(wx, 0., 0.))
    num_ps = len(ps_tags)

    embedded_ps_tags = []
    py = 0.; pz = 0.
    for p in embedded_nodes:
        try:
            py = p[1]
        except:
            pass    
        try:
            pz = p[2]
        except:
            pass
        embedded_ps_tags.append(_this.addPoint(p[0], py, pz))
    
    ## LINEs ##
    for it in range(num_ps - 1):
        ls_tags.append(_this.addLine(ps_tags[it], ps_tags[it+1]))
    ls_tags.append(_this.addLine(ps_tags[-1], ps_tags[0]))
    
    ## CURVEs (looped) ##
    curve_tags.append(_this.addCurveLoop(ls_tags))
    
    ## SURFACEs (PLANEs) ##
    plane_tags.append(_this.addPlaneSurface(curve_tags))

    # Set resolutions
    l_x = lx / res_x
    l_y = ly / res_y
    l_max = max(l_x, l_y)
    if el_size_max is not None:
        l_x = min(l_x, el_size_max)
        res_x = int(lx / l_x)
        l_y = min(l_y, el_size_max)
        res_y = int(ly / l_y)
    A = gmsh.model.mesh
    _this.synchronize() # Crucial to first call 'synchronize'.
    phy_g = gmsh.model.addPhysicalGroup(geo_dim, plane_tags)
    
    rx1 = int(res_x*wx/lx)
    rx2 = int(res_x*(1.-wx/lx))
    ry1 = int(res_y*(1.-wy/ly))
    ry2 = int(res_y*wy/ly)
    A.setTransfiniteCurve(ls_tags[0], res_y+1, coef=1)
    A.setTransfiniteCurve(ls_tags[1], res_x+1, coef=1)
    A.setTransfiniteCurve(ls_tags[2], ry2+1, coef=1)
    A.setTransfiniteCurve(ls_tags[3], rx2+1, coef=1)
    A.setTransfiniteCurve(ls_tags[4], ry1+1, coef=1)
    A.setTransfiniteCurve(ls_tags[5], rx1+1, coef=1)

    if el_size_max is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", el_size_max)
    if el_size_min is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", el_size_min)

    ## EMBEDED NODEs ##
    if len(embedded_ps_tags)>0:

        ### We need to force proper mesh sizes around embedded points
        
            ## Desired mesh size around any embedded nodes
            # VERSION-1 (better)
        A.generate(geo_dim)
        el_tags_of_embedded_nodes = [A.getElementsByCoordinates(n[0], n[1], 0)[0] for n in embedded_nodes]
        embedded_nodes_reses = (3.0 * A.getElementQualities(el_tags_of_embedded_nodes, 'volume')) ** 0.5 # ? Not clear whhy this gives best estimation of mesh size given element volume.
        A.clear()
            # VERSION-2 (sub-optimal)
        # factors_y = len(embedded_nodes) * [1.] # no effect of y-coordinates
        # # factors_y = [( 0.5 / scale - 1.) * (1. - abs(n[1] - ly/2.) / (ly/2.)) + 1. for n in embedded_nodes]
        # embedded_nodes_reses = [(scale ** ( 1. - abs(n[0]-lx/2) / (lx/2) ) ) * l_sides \
        #                         * factors_y[ni] for ni, n in enumerate(embedded_nodes)]
        
            ## Minimum distance of any other embedded node to each embedded node
        embedded_nodes_min_dists = [np.linalg.norm(n - np.delete(embedded_nodes, [ni], axis=0), axis = 1).min() for ni, n in enumerate(embedded_nodes)]
            
            ## Minimum distance among (minimum) distances of individual embedded nodes to the sides
        min_dists_to_sides = [min(n[0], lx - n[0]) for n in embedded_nodes]
        min_dist_to_sides = min(min_dists_to_sides)
        
            ## Set mesh fields
        A2 = A.field
        tag_f_thrs = [] # list of tags of all threshold fields
        for ip in range(len(embedded_ps_tags)):
            tag_f_d = A2.add('Distance')
            A2.setNumbers(tag_f_d, 'PointsList', [embedded_ps_tags[ip]])
            # Each threshold field
            tag_f_thr = A2.add('Threshold')
            A2.setNumber(tag_f_thr, 'InField', tag_f_d)
            A2.setNumber(tag_f_thr, 'SizeMin', embedded_nodes_reses[ip])
            A2.setNumber(tag_f_thr, 'DistMin', embedded_nodes_min_dists[ip])
            A2.setNumber(tag_f_thr, 'SizeMax', l_max) # the largest mesh size
            A2.setNumber(tag_f_thr, 'DistMax', min_dist_to_sides)
            tag_f_thrs.append(tag_f_thr)
        # Minimum field among all threshold fields defined above
        tag_f_min = A2.add('Min')
        A2.setNumbers(tag_f_min, 'FieldsList', tag_f_thrs)
        
            ## Set Background mesh based on minimum field established above
        A2.setAsBackgroundMesh(tag_f_min)

        A.embed(0, embedded_ps_tags, geo_dim, plane_tags[0])
    
    ## MESHING
    A.generate(geo_dim)
    gmsh.write(ff_msh)
    if write_geo:
        gmsh.write(ff_geo)
    gmsh.finalize()
        
    return ff_msh

def gmshAPI_slab2D_mesh(lx, ly, res_x, res_y, embedded_nodes, el_size_min=None, el_size_max=None \
                        , _path='./', _name='slab2D', write_geo=False):
    ## PARAMETERs
    geo_dim = 2

    ## FILEs
    make_path(_path)
    ff_geo = _path + '/' + _name + ".geo_unrolled" # if requested, writen after API.
    ff_msh = _path + '/' + _name + ".msh"

    ############### Gmsh API #################
    gmsh.initialize()
    model_tag = gmsh.model.add(_name)  # give the model a name
    ps_tags = []
    ls_tags = []
    curve_tags = []
    plane_tags = []

    _this = gmsh.model.occ
    # _this = gmsh.model.geo # Not sure, what is the difference to the above one!

    ps_tags.append(_this.addPoint(0., 0., 0.))
    ps_tags.append(_this.addPoint(lx, 0., 0.))
    ps_tags.append(_this.addPoint(lx, ly, 0.))
    ps_tags.append(_this.addPoint(0., ly, 0.))
    num_ps = len(ps_tags)

    embedded_ps_tags = []
    py = 0.; pz = 0.
    for p in embedded_nodes:
        try:
            py = p[1]
        except:
            pass    
        try:
            pz = p[2]
        except:
            pass
        embedded_ps_tags.append(_this.addPoint(p[0], py, pz))
    
    ## LINEs ##
    for it in range(num_ps - 1):
        ls_tags.append(_this.addLine(ps_tags[it], ps_tags[it+1]))
    ls_tags.append(_this.addLine(ps_tags[-1], ps_tags[0]))
    
    ## CURVEs (looped) ##
    curve_tags.append(_this.addCurveLoop(ls_tags))
    
    ## SURFACEs (PLANEs) ##
    plane_tags.append(_this.addPlaneSurface(curve_tags))

    # Set resolutions
    l_sides = ly / res_y
    if el_size_max is not None:
        l_sides = min(l_sides, el_size_max)
        res_y = int(ly / l_sides)
    A = gmsh.model.mesh
    _this.synchronize() # Crucial to first call 'synchronize'.
    phy_g = gmsh.model.addPhysicalGroup(geo_dim, plane_tags)
    
    A.setTransfiniteCurve(ls_tags[0], res_x+1, coef=1)
    A.setTransfiniteCurve(ls_tags[1], res_y+1, coef=1)
    A.setTransfiniteCurve(ls_tags[2], res_x+1, coef=1)
    A.setTransfiniteCurve(ls_tags[3], res_y+1, coef=1)

    if el_size_max is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", el_size_max)
    if el_size_min is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", el_size_min)

    ## EMBEDED NODEs ##
    if len(embedded_ps_tags)>0:

        ### We need to force proper mesh sizes around embedded points
        
            ## Desired mesh size around any embedded nodes
            # VERSION-1 (better)
        A.generate(geo_dim)
        el_tags_of_embedded_nodes = [A.getElementsByCoordinates(n[0], n[1], 0)[0] for n in embedded_nodes]
        embedded_nodes_reses = (3.0 * A.getElementQualities(el_tags_of_embedded_nodes, 'volume')) ** 0.5 # ? Not clear whhy this gives best estimation of mesh size given element volume.
        A.clear()
            # VERSION-2 (sub-optimal)
        # factors_y = len(embedded_nodes) * [1.] # no effect of y-coordinates
        # # factors_y = [( 0.5 / scale - 1.) * (1. - abs(n[1] - ly/2.) / (ly/2.)) + 1. for n in embedded_nodes]
        # embedded_nodes_reses = [(scale ** ( 1. - abs(n[0]-lx/2) / (lx/2) ) ) * l_sides \
        #                         * factors_y[ni] for ni, n in enumerate(embedded_nodes)]
        
            ## Minimum distance of any other embedded node to each embedded node
        embedded_nodes_min_dists = [np.linalg.norm(n - np.delete(embedded_nodes, [ni], axis=0), axis = 1).min() for ni, n in enumerate(embedded_nodes)]
            
            ## Minimum distance among (minimum) distances of individual embedded nodes to the sides
        min_dists_to_sides = [min(n[0], lx - n[0]) for n in embedded_nodes]
        min_dist_to_sides = min(min_dists_to_sides)
        
            ## Set mesh fields
        A2 = A.field
        tag_f_thrs = [] # list of tags of all threshold fields
        for ip in range(len(embedded_ps_tags)):
            tag_f_d = A2.add('Distance')
            A2.setNumbers(tag_f_d, 'PointsList', [embedded_ps_tags[ip]])
            # Each threshold field
            tag_f_thr = A2.add('Threshold')
            A2.setNumber(tag_f_thr, 'InField', tag_f_d)
            A2.setNumber(tag_f_thr, 'SizeMin', embedded_nodes_reses[ip])
            A2.setNumber(tag_f_thr, 'DistMin', embedded_nodes_min_dists[ip])
            A2.setNumber(tag_f_thr, 'SizeMax', l_sides) # the largest mesh size (at sides)
            A2.setNumber(tag_f_thr, 'DistMax', min_dist_to_sides)
            tag_f_thrs.append(tag_f_thr)
        # Minimum field among all threshold fields defined above
        tag_f_min = A2.add('Min')
        A2.setNumbers(tag_f_min, 'FieldsList', tag_f_thrs)
        
            ## Set Background mesh based on minimum field established above
        A2.setAsBackgroundMesh(tag_f_min)

        A.embed(0, embedded_ps_tags, geo_dim, plane_tags[0])
    
    ## MESHING
    A.generate(geo_dim)
    gmsh.write(ff_msh)
    if write_geo:
        gmsh.write(ff_geo)
    gmsh.finalize()
        
    return ff_msh

def gmshAPI_notched_rectangle_mesh(lx, ly, l_notch, h_notch, c_notch=None \
                           , load_Xrange=None, left_sup=None, right_sup=None, left_sup_w=0., right_sup_w=0. \
                           , res_y=3, scale=0.5, embedded_nodes=[], el_size_min=None, el_size_max=None \
                           , _path='./', _name='notched_rectangle', write_geo=False):
    """
    Generate a mesh (an *.msh file) for this geometry:
    __________________________
    |                        |
    |           __           |
    |__________| |___________|
    
    left_sup = the x coordinate at which the left support is located (y=0)
    right_sup = the x coordinate at which the right support is located (y=0)
    c_notch is the central vertical axis of the notch.
        (when is None, it will be set to lx/2 by defaultthe, meaning the middle.)
    """
    ## PARAMETERs
    geo_dim = 2
    if c_notch is None:
        c_notch = lx/2
    if load_Xrange is None:
        load_Xrange = [lx/3, 2*lx/3]
    load_range = load_Xrange[1] - load_Xrange[0]
    if load_range > lx * 1.0e-4:
        _shift_load = 1
    else:
        _shift_load = 0
        load_range = 0.
    if left_sup is None:
        left_sup = 0.
    assert(0. <= left_sup < lx / 2.)
    assert(0. <= left_sup - left_sup_w / 2.)
    if right_sup is None:
        right_sup = lx
    assert(lx / 2 < right_sup <= lx)
    assert(right_sup + right_sup_w / 2. <= lx)
        
    ## FILEs
    make_path(_path)
    ff_geo = _path + '/' + _name + ".geo_unrolled" # if requested, writen after API.
    ff_msh = _path + '/' + _name + ".msh"
    
    ############### Gmsh API #################
    gmsh.initialize()
    model_tag = gmsh.model.add(_name)  # give the model a name
    ps_tags = []
    ls_tags = []
    curve_tags = []
    plane_tags = []
    
    _this = gmsh.model.occ
    # _this = gmsh.model.geo # Not sure, what is the difference to the above one!
    
    ## POINTs ##
    shift_notch = 1; b_notch = False
    ps_tags.append(_this.addPoint(-l_notch/2. + c_notch, 0, 0))
    if abs(l_notch) > lx * 1e-8: # We have notch
        ps_tags.append(_this.addPoint(-l_notch/2. + c_notch, h_notch, 0))
        ps_tags.append(_this.addPoint(l_notch/2. + c_notch, h_notch, 0))
        ps_tags.append(_this.addPoint(l_notch/2. + c_notch, 0, 0))
        shift_notch = 4; b_notch = True
    shift = 0
    xx1 = right_sup - right_sup_w / 2.
    xx2 = xx1 + right_sup_w
    if xx1!=lx:
        ps_tags.append(_this.addPoint(xx1, 0, 0))
        shift += 1
    if xx2!=lx and xx2!=xx1:
        ps_tags.append(_this.addPoint(xx2, 0, 0))
        shift += 1
    ps_tags.append(_this.addPoint(lx, 0, 0))
    ps_tags.append(_this.addPoint(lx, ly, 0))
    ps_tags.append(_this.addPoint(load_Xrange[1], ly, 0))
    if _shift_load:
        ps_tags.append(_this.addPoint(load_Xrange[0], ly, 0))
    ps_tags.append(_this.addPoint(0, ly, 0))
    ps_tags.append(_this.addPoint(0, 0, 0))
    shift_l = 0
    xx1 = left_sup - left_sup_w / 2.
    xx2 = xx1 + left_sup_w
    if xx1!=0.:
        ps_tags.append(_this.addPoint(xx1, 0, 0))
        shift_l += 1
    if xx2!=0. and xx2!=xx1:
        ps_tags.append(_this.addPoint(xx2, 0, 0))
        shift_l += 1
    num_ps = len(ps_tags)
    
    embedded_ps_tags = []
    py = 0.; pz = 0.
    for p in embedded_nodes:
        try:
            py = p[1]
        except:
            pass    
        try:
            pz = p[2]
        except:
            pass
        embedded_ps_tags.append(_this.addPoint(p[0], py, pz))
    
    ## LINEs ##
    for it in range(num_ps - 1):
        ls_tags.append(_this.addLine(ps_tags[it], ps_tags[it+1]))
    ls_tags.append(_this.addLine(ps_tags[-1], ps_tags[0]))
    
    ## CURVEs (looped) ##
    curve_tags.append(_this.addCurveLoop(ls_tags))
    
    ## SURFACEs (PLANEs) ##
    plane_tags.append(_this.addPlaneSurface(curve_tags))
    
    ## MESH RESULUTIONs ##
    # Set coefficients
    # Vertical sides
    l_sides = ly / res_y
    if el_size_max is not None:
        l_sides = min(l_sides, el_size_max)
        res_y = int(ly / l_sides)
    # Top sides (and possible load_range)
    l_top_left = load_Xrange[0] - 0.
    n_top_left, r_top_left, l0_top_left = geometric_resolution_over_length(l=l_top_left, scale=scale, l0=l_sides)
    l_top_right = lx - load_Xrange[1]
    n_top_right, r_top_right, l0_top_right = geometric_resolution_over_length(l=l_top_right, scale=scale, l0=l_sides)
    if _shift_load:
        ln_top_sides = scale * (l0_top_left + l0_top_right) / 2.
        n_load_range = max(1, round(load_range / ln_top_sides))
    
    # Bottom left side (and possible left support nodes)
    l_left = c_notch - l_notch / 2. # total length
    # Potentially we have three segments (the first and second one can be of zero-length)
    l1, l2 = left_sup - left_sup_w / 2., left_sup_w
    l3 = l_left - (l1 + l2)
    scale_l1, scale_l2, scale_l3 = [scale ** (aa / l_left) for aa in [l1, l2, l3]]
    n_l1, r_l1, l0_l1 = geometric_resolution_over_length(l=l1, scale=scale_l1, l0=l_sides)
    l0_l2 = scale_l1 * l0_l1
    n_l2, r_l2, l0_l2 = geometric_resolution_over_length(l=l2, scale=scale_l2, l0=l0_l2)
    l0_l3 = scale_l2 * l0_l2
    n_l3, r_l3, l0_l3 = geometric_resolution_over_length(l=l3, scale=scale_l3, l0=l0_l3)
    ln_l3 = scale_l3 * l0_l3
    
    # Bottom right side (and possible right support)
    l_right = (lx - c_notch) - l_notch / 2. # total length
    # Potentially we have three segments (the first and second one can be of zero-length)
    r1, r2 = lx - (right_sup + right_sup_w / 2.), right_sup_w
    r3 = l_right - (r1 + r2)
    scale_r1, scale_r2, scale_r3 = [scale ** (aa / l_right) for aa in [r1, r2, r3]]
    n_r1, r_r1, l0_r1 = geometric_resolution_over_length(l=r1, scale=scale_r1, l0=l_sides)
    l0_r2 = scale_r1 * l0_r1
    n_r2, r_r2, l0_r2 = geometric_resolution_over_length(l=r2, scale=scale_r2, l0=l0_r2)
    l0_r3 = scale_r2 * l0_r2
    n_r3, r_r3, l0_r3 = geometric_resolution_over_length(l=r3, scale=scale_r3, l0=l0_r3)
    ln_r3 = scale_r3 * l0_r3
    
    # Notch
    l0_notch = (ln_l3 + ln_r3) / 2.
    n_h_notch = max(2, round(h_notch / l0_notch))
    n_l_notch = max(2, round(l_notch / l0_notch))
    
    # Set resolutions
    A = gmsh.model.mesh
    _this.synchronize() # Crucial to first call 'synchronize'.
    phy_g = gmsh.model.addPhysicalGroup(geo_dim, plane_tags)
    
    if b_notch:
        # Notch hight
        A.setTransfiniteCurve(ls_tags[0], n_h_notch+1, coef=1)
        A.setTransfiniteCurve(ls_tags[2], n_h_notch+1, coef=1)
        # Notch width (length)
        A.setTransfiniteCurve(ls_tags[1], n_l_notch+1, coef=1)
    # Top
    A.setTransfiniteCurve(ls_tags[1+shift_notch+shift], n_top_right+1, coef=r_top_right)
    if _shift_load:
        A.setTransfiniteCurve(ls_tags[1+shift_notch+shift+_shift_load], n_load_range+1, coef=1, meshType='Bump') # Bump means concentrating elements in the middle.
    A.setTransfiniteCurve(ls_tags[2+shift_notch+shift+_shift_load], n_top_left+1, coef=-r_top_left)
    # Sides
    A.setTransfiniteCurve(ls_tags[shift_notch+shift], res_y+1, coef=1)
    A.setTransfiniteCurve(ls_tags[3+shift_notch+shift+_shift_load], res_y+1, coef=1)
    # Bottom-right
    A.setTransfiniteCurve(ls_tags[shift_notch-1], n_r3+1, coef=-r_r3)
    if r2!=0.:
        A.setTransfiniteCurve(ls_tags[shift_notch], n_r2+1, coef=-r_r2)
    if r1!=0.:
        A.setTransfiniteCurve(ls_tags[shift_notch-1+shift], n_r1+1, coef=-r_r1)
    ## Bottom-left
    if l1!=0.:
        A.setTransfiniteCurve(ls_tags[4+shift_notch+shift+_shift_load], n_l1+1, coef=r_l1)
    if l2!=0.:
        A.setTransfiniteCurve(ls_tags[4+shift_notch+shift+_shift_load+shift_l-1], n_l2+1, coef=r_l2)
    A.setTransfiniteCurve(ls_tags[4+shift_notch+shift+_shift_load+shift_l], n_l3+1, coef=r_l3)
    
    if el_size_max is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", el_size_max)
    if el_size_min is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", el_size_min)
    
    ## EMBEDED NODEs ##
    if len(embedded_ps_tags)>0:
        # gmsh.option.setNumber("Mesh.Algorithm", 5)
            # It is mentioned that: algorithm 6 (default?) is most advanced, but 5 is more suitable for complex geometries.
        
        ### We need to force proper mesh sizes around embedded points
        
            ## Desired mesh size around any embedded nodes
            # VERSION-1 (better)
        A.generate(geo_dim)
        el_tags_of_embedded_nodes = [A.getElementsByCoordinates(n[0], n[1], 0)[0] for n in embedded_nodes]
        embedded_nodes_reses = (3.0 * A.getElementQualities(el_tags_of_embedded_nodes, 'volume')) ** 0.5 # ? Not clear whhy this gives best estimation of mesh size given element volume.
        A.clear()
            # VERSION-2 (sub-optimal)
        # factors_y = len(embedded_nodes) * [1.] # no effect of y-coordinates
        # # factors_y = [( 0.5 / scale - 1.) * (1. - abs(n[1] - ly/2.) / (ly/2.)) + 1. for n in embedded_nodes]
        # embedded_nodes_reses = [(scale ** ( 1. - abs(n[0]-lx/2) / (lx/2) ) ) * l_sides \
        #                         * factors_y[ni] for ni, n in enumerate(embedded_nodes)]
        
            ## Minimum distance of any other embedded node to each embedded node
        embedded_nodes_min_dists = [np.linalg.norm(n - np.delete(embedded_nodes, [ni], axis=0), axis = 1).min() for ni, n in enumerate(embedded_nodes)]
            
            ## Minimum distance among (minimum) distances of individual embedded nodes to the sides
        min_dists_to_sides = [min(n[0], lx - n[0]) for n in embedded_nodes]
        min_dist_to_sides = min(min_dists_to_sides)
        
            ## Set mesh fields
        A2 = A.field
        tag_f_thrs = [] # list of tags of all threshold fields
        for ip in range(len(embedded_ps_tags)):
            tag_f_d = A2.add('Distance')
            A2.setNumbers(tag_f_d, 'PointsList', [embedded_ps_tags[ip]])
            # Each threshold field
            tag_f_thr = A2.add('Threshold')
            A2.setNumber(tag_f_thr, 'InField', tag_f_d)
            A2.setNumber(tag_f_thr, 'SizeMin', embedded_nodes_reses[ip])
            A2.setNumber(tag_f_thr, 'DistMin', embedded_nodes_min_dists[ip])
            A2.setNumber(tag_f_thr, 'SizeMax', l_sides) # the largest mesh size (at sides)
            A2.setNumber(tag_f_thr, 'DistMax', min_dist_to_sides)
            tag_f_thrs.append(tag_f_thr)
        # Minimum field among all threshold fields defined above
        tag_f_min = A2.add('Min')
        A2.setNumbers(tag_f_min, 'FieldsList', tag_f_thrs)
        
            ## Set Background mesh based on minimum field established above
        A2.setAsBackgroundMesh(tag_f_min)
        
        ### Set embedded nodes
        A.embed(0, embedded_ps_tags, geo_dim, plane_tags[0])
    
    ## MESHING
    A.generate(geo_dim)
    gmsh.write(ff_msh)
    if write_geo:
        gmsh.write(ff_geo)
    gmsh.finalize()
        
    return ff_msh

##############################################################################
##############################################################################
###################  OLD STUFF (by building a *.geo file) ####################
##############################################################################
##############################################################################

def notched_rectangle_mesh_by_geo(lx, ly, l_notch, h_notch, c_notch=None \
                           , load_Xrange=None, left_sup=None, right_sup=None \
                           , res_x = 30, res_y = 3, res_notch_l = 3, res_notch_h = 4 \
                           , _path='./', _name='notched_rectangle'):
    """
    Returns a FEniCS mesh object for the geometry defined in "notched_rectangle_geo" (See above)
    """
    notched_rectangle_geo(lx, ly, l_notch, h_notch, c_notch=c_notch \
                          , load_Xrange=load_Xrange, left_sup=left_sup, right_sup=right_sup \
                              , res_x=res_x, res_y=res_y, res_notch_l=res_notch_l, res_notch_h=res_notch_h \
                              , _path=_path, _name=_name)
    
    ## FILEs
    f_geo = _path + _name + ".geo"
    f_msh = _path + _name + ".msh"
    f_xdmf = _path + _name + ".xdmf"
    
    ## Convert *.geo to *.msh (Nodal coordinates have Z-entries all being ZERO)
    from subprocess import call
    call(["gmsh", "-2", f_geo, "-o", f_msh])
    
    ## Convert *.msh to *.xdmf
        # WAY-1 (directly in python)
    import meshio
    meshio_mesh = meshio.read(f_msh) # This mesh has nodal coordinates in XYZ (all Z entries being zero).
    meshio_mesh_2D = meshio.Mesh(points=meshio_mesh.points[:,:2], cells=meshio_mesh.cells) # To have coordinates of only XY dimensions.
    meshio.write(f_xdmf, meshio_mesh_2D)
        # WAY-2
    # call(["meshio-convert", f_msh, f_xdmf, "-p", "-z"]) # Those flags are neede to get an *.xdmf mesh with 2-D coordinates.
    
    ## Read *.xdmf mesh into FEniCS mesh
    import dolfin as df
    mesh = df.Mesh()
    with df.XDMFFile(f_xdmf) as ff:
        ff.read(mesh)
    return mesh

def notched_rectangle_geo(lx, ly, l_notch, h_notch, c_notch=None \
                          , load_Xrange=None, left_sup=None, right_sup=None \
                              , res_x = 30, res_y = 3, res_notch_l = 3, res_notch_h = 4 \
                                  , _path='./', _name='notched_rectangle'):
    """
    Generate a *.geo file for this geometry:
    __________________________
    |                        |
    |           __           |
    |__________| |___________|
    
    left_sup = the x coordinate at which the left support is located (y=0)
    right_sup = the x coordinate at which the right support is located (y=0)
    c_notch is the central vertical axis of the notch.
        (when is None, it will be set to lx/2 by defaultthe, meaning the middle.)
    """
    res_half_x = np.ceil(res_x / 2) - 2
    if c_notch is None:
        c_notch = lx/2
    if load_Xrange is None:
        load_Xrange = [lx/3, 2*lx/3]
    res_x_mid = np.ceil(1.15 * (load_Xrange[1] - load_Xrange[0]) * res_x / lx)
    res_x_sides = np.ceil((res_x - res_x_mid) / 2 / 1.15)
    if ((left_sup is None) and (right_sup is None)):
        # The supports are supposed at the two ends.
        # ---> No need for extra points for supports.
        with open(_path + _name + ".geo", "w") as f:
            f.write("\nL = " + str(lx) + ";")
            f.write("\nH = " + str(ly) + ";")
            f.write("\nLnotch = " + str(l_notch) + ";")
            f.write("\nHnotch = " + str(h_notch) + ";")
            f.write("\nCnotch = " + str(c_notch) + ";")
            f.write("\nXl = " + str(load_Xrange[0]) + ";")
            f.write("\nXr = " + str(load_Xrange[1]) + ";")
            
            f.write("\nPoint(1) = {-Lnotch/2 + Cnotch, 0, 0, 1.0};")
            f.write("\nPoint(2) = {-Lnotch/2 + Cnotch, Hnotch, 0, 1.0};")
            f.write("\nPoint(3) = {Lnotch/2 + Cnotch, Hnotch, 0, 1.0};")
            f.write("\nPoint(4) = {Lnotch/2 + Cnotch, 0, 0, 1.0};")
            f.write("\nPoint(5) = {L, 0, 0, 1.0};")
            f.write("\nPoint(6) = {L, H, 0, 1.0};")
            f.write("\nPoint(7) = {Xr, H, 0, 1.0};")
            f.write("\nPoint(8) = {Xl, H, 0, 1.0};")
            f.write("\nPoint(9) = {0, H, 0, 1.0};")
            f.write("\nPoint(10) = {0, 0, 0, 1.0};")
            
            # define Lines
            n_points = 10
            for i in range(n_points-1):
                f.write("\nLine(" + str(i+1) + ") = {" + str(i+1) + ", " + str(i+2) + "};")
            f.write("\nLine(" + str(n_points) + ") = {" + str(n_points) + ", 1};")
            
            # define Loop
            _loop_nums = '1'
            for i in range(2, n_points+1):
                _loop_nums += ', ' + str(i)
            f.write("\nCurve Loop(1) = {" + _loop_nums + "};")
            
            f.write("\nPlane Surface(1) = {1};")
            f.write("\nPhysical Surface(1) = {1};")
    
            f.write("\n// notch height")
            f.write("\nTransfinite Curve {1, 3} = " + str(res_notch_h + 1) + " Using Progression 1;")
            f.write("\n// notch width")
            f.write("\nTransfinite Curve {2} = " + str(res_notch_l + 1) + " Using Progression 1;")
            
            f.write("\n// top lines. Bump means concentrating elements in the middle of the line.")
            f.write("\nTransfinite Curve {-6} = " + str(res_x_sides) + " Using Progression 1.15;")
            f.write("\nTransfinite Curve {7} = " + str(res_x_mid+2) + " Using Bump 2;")
            f.write("\nTransfinite Curve {8} = " + str(res_x_sides) + " Using Progression 1.15;")
            
            f.write("\n// side lines")
            f.write("\nTransfinite Curve {9, 5} = " + str(res_y + 1) + " Using Progression 1;")
            
            f.write("\n// bot right. Notice the -8 to indicate the direction of the progression")
            f.write("\nTransfinite Curve {-10} = " + str(res_half_x) + " Using Progression 1.15;")
            f.write("\nTransfinite Curve {4} = " + str(res_half_x) + " Using Progression 1.15;")
    else:
        assert (left_sup>0) and (right_sup<lx)
        with open(_path + _name + ".geo", "w") as f:
            f.write("\nL = " + str(lx) + ";")
            f.write("\nH = " + str(ly) + ";")
            f.write("\nLnotch = " + str(l_notch) + ";")
            f.write("\nHnotch = " + str(h_notch) + ";")
            f.write("\nCnotch = " + str(c_notch) + ";")
            f.write("\nXl = " + str(load_Xrange[0]) + ";")
            f.write("\nXr = " + str(load_Xrange[1]) + ";")
            f.write("\nl_sup = " + str(left_sup) + ";")
            f.write("\nr_sup = " + str(right_sup) + ";")
            
            f.write("\nPoint(1) = {-Lnotch/2 + Cnotch, 0, 0, 1.0};")
            f.write("\nPoint(2) = {-Lnotch/2 + Cnotch, Hnotch, 0, 1.0};")
            f.write("\nPoint(3) = {Lnotch/2 + Cnotch, Hnotch, 0, 1.0};")
            f.write("\nPoint(4) = {Lnotch/2 + Cnotch, 0, 0, 1.0};")
            f.write("\nPoint(5) = {r_sup, 0, 0, 1.0};")
            f.write("\nPoint(6) = {L, 0, 0, 1.0};")
            f.write("\nPoint(7) = {L, H, 0, 1.0};")
            f.write("\nPoint(8) = {Xr, H, 0, 1.0};")
            f.write("\nPoint(9) = {Xl, H, 0, 1.0};")
            f.write("\nPoint(10) = {0, H, 0, 1.0};")
            f.write("\nPoint(11) = {0, 0, 0, 1.0};")
            f.write("\nPoint(12) = {l_sup, 0, 0, 1.0};")
            
            # define Lines
            n_points = 12
            for i in range(n_points-1):
                f.write("\nLine(" + str(i+1) + ") = {" + str(i+1) + ", " + str(i+2) + "};")
            f.write("\nLine(" + str(n_points) + ") = {" + str(n_points) + ", 1};")
            
            # define Loop
            _loop_nums = '1'
            for i in range(2, n_points+1):
                _loop_nums += ', ' + str(i)
            f.write("\nCurve Loop(1) = {" + _loop_nums + "};")
            
            f.write("\nPlane Surface(1) = {1};")
            f.write("\nPhysical Surface(1) = {1};")
    
            f.write("\n// notch height")
            f.write("\nTransfinite Curve {1, 3} = " + str(res_notch_h + 1) + " Using Progression 1;")
            f.write("\n// notch width")
            f.write("\nTransfinite Curve {2} = " + str(res_notch_l + 1) + " Using Progression 1;")
            
            f.write("\n// top line. Bump means concentrating elements in the middle of the line.")
            f.write("\nTransfinite Curve {-7} = " + str(res_x_sides) + " Using Progression 1.15;")
            f.write("\nTransfinite Curve {8} = " + str(res_x_mid+2) + " Using Bump 2;")
            f.write("\nTransfinite Curve {9} = " + str(res_x_sides) + " Using Progression 1.15;")
            
            f.write("\n// side lines")
            f.write("\nTransfinite Curve {10, 6} = " + str(res_y + 1) + " Using Progression 1;")
            
            f.write("\n// bot right. Notice the minus sign to indicate the direction of the progression")
            ll = 0.5 * (lx - l_notch)
            nl1 = np.ceil(res_half_x * left_sup / ll)
            nl2 = np.ceil(res_half_x * (ll - left_sup) / ll)
            aa = 1.15 ** nl1; bb = 1.15 ** nl2
            nl1 = max(1, np.ceil(res_half_x * aa / (aa + bb)) - 1)
            nl2 = np.ceil(res_half_x * bb / (aa + bb))
            f.write("\nTransfinite Curve {-11} = " + str(nl1) + " Using Progression 1.15;")
            f.write("\nTransfinite Curve {-12} = " + str(nl2) + " Using Progression 1.15;")
            right_sup = lx - right_sup
            nr1 = np.ceil(res_half_x * right_sup / ll)
            nr2 = np.ceil(res_half_x * (ll - right_sup) / ll)
            aa = 1.15 ** nr1; bb = 1.15 ** nr2
            nr1 = max(1, np.ceil(res_half_x * aa / (aa + bb)) - 1)
            nr2 = np.ceil(res_half_x * bb / (aa + bb))
            f.write("\nTransfinite Curve {5} = " + str(nr1) + " Using Progression 1.15;")
            f.write("\nTransfinite Curve {4} = " + str(nr2) + " Using Progression 1.15;")
    