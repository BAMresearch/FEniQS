from feniQS.structure.helper_mesh_gmsh_meshio import *
import dolfin as df

pth_helper_mesh_fenics = CollectPaths('helper_mesh_fenics.py')
pth_helper_mesh_fenics.add_script(pth_helper_mesh_gmsh_meshio)

def bcc_mesh_parametric(r_strut, l_rve, n_rve, l_cell, add_plates=True, shape_name="bcc" \
             , _path='./', _name='parametric_bcc'):
    ff_msh = gmshAPI_generate_bcc_lattice(r_strut=r_strut, l_rve=l_rve, n_rve=n_rve \
                                           , l_cell=l_cell, add_plates=add_plates, shape_name=shape_name \
                                           , _path=_path, _name=_name)
    ff_xdmf = meshio_get_xdmf_from_msh(ff_msh, path_xdmf=_path, f_xdmf=_name+'.xdmf', geo_dim=3)
    mesh = df.Mesh()
    with df.XDMFFile(ff_xdmf) as ff:
        ff.read(mesh)
    return mesh

def slab2D_mesh(lx, ly, res_x, res_y, embedded_nodes, _path, _name='slab2D'):
    ## Use Gmsh-API and meshio
    ff_msh = gmshAPI_slab2D_mesh(lx=lx, ly=ly, res_x=res_x, res_y=res_y \
                                 , embedded_nodes=embedded_nodes, _path=_path, _name=_name)
    ff_xdmf = meshio_get_xdmf_from_msh(ff_msh, path_xdmf=_path, f_xdmf=_name+'.xdmf', geo_dim=2)
    mesh = df.Mesh()
    with df.XDMFFile(ff_xdmf) as ff:
        ff.read(mesh)
    
    ##### CRUCIAL #####
    # We modify values of the given embedded_nodes to the exact nodal coordinates of the generated mesh.
    # ---> Some veryyyy small deviation can emerge after calling methods 'gmshAPI_notched_rectangle_mesh'
    #      and 'meshio_get_xdmf_from_msh' and reading XDMF file to a FEniCS mesh.
    #      Such deviation might cause issue: A given embedded node might get outside of the generated FEniCS mesh !
    tol = mesh.rmin() / 1000.
    for ie, ce in enumerate(embedded_nodes):
        for cm in mesh.coordinates():
            if np.linalg.norm(ce - cm) < tol:
                embedded_nodes[ie, :] = cm[:]
    
    return mesh

def notched_rectangle_mesh(lx, ly, l_notch, h_notch, c_notch=None \
                           , load_Xrange=None, left_sup=None, right_sup=None, left_sup_w=0., right_sup_w=0. \
                           , res_y=3, scale=0.5, embedded_nodes=[], el_size_min=None, el_size_max=None \
                           , _path='./', _name='notched_rectangle', write_geo=False):
    """
    Generate a fenics mesh (through *.msh and *.xdmf files) for this geometry:
    __________________________
    |                        |
    |           __           |
    |__________| |___________|
    
    left_sup = the x coordinate at which the left support is located (y=0)
    right_sup = the x coordinate at which the right support is located (y=0)
    c_notch is the central vertical axis of the notch.
        (when is None, it will be set to lx/2 by defaultthe, meaning the middle.)
    """
    ## Use Gmsh-API and meshio
    ff_msh = gmshAPI_notched_rectangle_mesh(lx, ly, l_notch, h_notch, c_notch=c_notch \
                            , load_Xrange=load_Xrange, left_sup=left_sup, right_sup=right_sup, left_sup_w=left_sup_w, right_sup_w=right_sup_w \
                            , res_y=res_y, scale=scale \
                            , embedded_nodes=embedded_nodes, el_size_min=el_size_min, el_size_max=el_size_max \
                            , _path=_path, _name=_name, write_geo=write_geo)
    ff_xdmf = meshio_get_xdmf_from_msh(ff_msh, path_xdmf=_path, f_xdmf=_name+'.xdmf', geo_dim=2)
    mesh = df.Mesh()
    with df.XDMFFile(ff_xdmf) as ff:
        ff.read(mesh)
    
    ##### CRUCIAL #####
    # We modify values of the given embedded_nodes to the exact nodal coordinates of the generated mesh.
    # ---> Some veryyyy small deviation can emerge after calling methods 'gmshAPI_notched_rectangle_mesh'
    #      and 'meshio_get_xdmf_from_msh' and reading XDMF file to a FEniCS mesh.
    #      Such deviation might cause issue: A given embedded node might get outside of the generated FEniCS mesh !
    tol = mesh.rmin() / 1000.
    for ie, ce in enumerate(embedded_nodes):
        for cm in mesh.coordinates():
            if np.linalg.norm(ce - cm) < tol:
                embedded_nodes[ie, :] = cm[:]
    
    return mesh

def refined_rectangle_mesh(lx, ly, x_from, x_to, unit_res, refined_s):
    """
    unit_res: the number of elements per unit length of the geometry (can be any float number)
    refined_s: the size of refined mesh at the zone between x_from to x_to
    """
    tol = 1e-12
    mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(lx, ly), int(unit_res * lx), int(unit_res * ly), diagonal='right')
    def middle_part(x):
        return df.between(x[0], (x_from - tol, x_to + tol))
    dom = df.AutoSubDomain(middle_part)
    s = max(lx/int(unit_res * lx), ly/int(unit_res * ly))
    factr = s / refined_s
    import math
    ref_times = math.ceil(math.log2(factr))
    for i in range(ref_times):
        mf_bool = df.MeshFunction('bool', mesh, 2)
        mf_bool.set_all(False)
        dom.mark(mf_bool, True)
        mesh = df.refine(mesh, mf_bool)
    return mesh

def mesh_cells_diameters_over_domain(mesh, dom):
    assert isinstance(dom, df.SubDomain)
    _dim = mesh.ufl_cell().topological_dimension()
    mf_size = df.MeshFunction("size_t", mesh, _dim, 0)
    mf_size.set_all(0)
    dom.mark(mf_size, 1)
    marked_cells = df.SubsetIterator(mf_size, 1) # cells of subdomains with marked value of 1
    crad = df.Circumradius(mesh)
    cells_diameters = df.project(crad, df.FunctionSpace(mesh, "DG", 0)).vector()[:] * 2 # for all cells
    return [ cells_diameters[cell.global_index()] for cell in marked_cells ]

def refine_domain_of_mesh_once(mesh, dom):
    assert isinstance(dom, df.SubDomain)
    _dim = mesh.ufl_cell().topological_dimension()
    mf_bool = df.MeshFunction('bool', mesh, _dim, 0)
    mf_bool.set_all(0)
    dom.mark(mf_bool, 1)
    return df.refine(mesh, mf_bool) # a new mesh object

def refine_domain_of_mesh(mesh, dom, min_s, max_refs=10, _check=True):
    """
    mesh: original mesh
    dom: the domain over which the refinement is applied.
    min_s: the minimum required size of refined mesh at the specified domain.
    It returns the desired refined mesh.
    """
    assert isinstance(dom, df.SubDomain)
    dom_cells_diameters = mesh_cells_diameters_over_domain(mesh, dom)
    s = max(dom_cells_diameters)
    
    ## WAY-1: more conservative and robust
    _ref=0
    while s>min_s and _ref<max_refs:
        mesh = refine_domain_of_mesh_once(mesh, dom)
        dom_cells_diameters = mesh_cells_diameters_over_domain(mesh, dom)
        s = max(dom_cells_diameters)
        _ref += 1
    ## WAY-2: not working well (the below check will strangly fail in some cases !)
    # factr = s / min_s
    # import math
    # ref_times = math.ceil(math.log2(factr))
    # for i in range(ref_times):
    #     mesh = refine_domain_of_mesh_once(mesh, dom)
    if _check:
        dom_cells_diameters = mesh_cells_diameters_over_domain(mesh, dom)
        if max(dom_cells_diameters) > min_s:
            raise AssertionError(f"Maximum diameter of some cells in the specified domain exceeds the desired {min_s}.")
    return mesh
