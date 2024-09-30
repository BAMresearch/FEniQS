import numpy as np
import meshio
import gmsh
import os

from feniQS.general.general import make_path, CollectPaths

pth_helper_mesh_gmsh_meshio = CollectPaths('./feniQS/structure/helper_mesh_gmsh_meshio.py')

"""
f: file (only file name + format)
ff: full file (path + file name + format)
"""

def extrude_triangled_mesh_by_meshio(ff_mesh_surface, ff_mesh_extruded, dz, res=1):
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
    """
    mesh = meshio.read(ff_mesh_surface)
    if any([c.type!='triangle' for c in mesh.cells]):
        raise ValueError(f"The input mesh must only contain triangle elements.")
    ps = mesh.points
    n_ps = ps.shape[0]
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
    
    for i, dz_i in enumerate(dzs):
        for i_side, dz_side in enumerate(dz_i):
            for iz in range(1, res[i_side]+1):
                shift = shift_nodes_side[i_side] + iz * n_ps
                if isinstance(dz_side, (int, float,)):
                    ps_3d[i+shift, :2] = ps3[i,:2]
                    ps_3d[i+shift, 2] = iz*dz_side/res[i_side]
                else:
                    ps_3d[i+shift,:] = ps3[i,:] + iz*dz_side/res[i_side]
    mesh.points = ps_3d
    mesh.cells[0].type = 'tetra'
    mesh.cells[0].dim = 3
    
    mesh_2d_cell_data = mesh.cells[0].data
    n_cells_2d = mesh_2d_cell_data.shape[0]
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
    
    meshio.write(ff_mesh_extruded, mesh)

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
    