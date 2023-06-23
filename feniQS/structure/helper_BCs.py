import numpy as np
import dolfin as df
from feniQS.fenics_helpers.fenics_functions import boundary_condition_pointwise, boundary_condition, pth_fenics_functions

from feniQS.general.general import CollectPaths
pth_helper_BCs = CollectPaths('./feniQS/structure/helper_BCs.py')
pth_helper_BCs.add_script(pth_fenics_functions)

def bc_on_middle_3point_bending(x_from, x_to, ly, i_u, u_expr, tol):
    if abs(x_from - x_to) > tol:
        ## on an interval
        def middle_top(x, on_boundary):
            return on_boundary and df.between(x[0], (x_from - tol, x_to + tol)) and df.near(x[1], ly, tol)
        bc_middle_top, bc_middle_top_dofs = boundary_condition(i_u.sub(1), u_expr, middle_top)
    else:
        ## on a single point
        def middle_top(x, on_boundary):
            return df.near(x[0], x_from, tol) and df.near(x[1], ly, tol)
        bc_middle_top, bc_middle_top_dofs = boundary_condition_pointwise(i_u.sub(1), u_expr, middle_top)
    
    if len(bc_middle_top_dofs) == 0:
        raise ValueError('No DOFs were found for the middle displacement loading. You might need to redefine the mesh.')
    return bc_middle_top, bc_middle_top_dofs, middle_top

def load_and_bcs_on_3point_bending(mesh, lx, ly, x_from, x_to, i_u, u_expr \
                                   , left_sup=0., right_sup=None, left_sup_w=0., right_sup_w=0., x_fix='left'):
    """
    mesh: a rectangle mesh of lx*ly
    i_u: function_space of the full displacement field
    u_expr: an expression for the displacement load
    
    For a 3point-bending structure ready for a modelling (e.g. to be given a material type in the future)
    , returns:
        bcs_DR, bcs_DR_inhom
    """
    assert(mesh.geometric_dimension()==2)
    assert(i_u.num_sub_spaces()==2)
    tol = mesh.rmin() / 1000.
    bcs_DR = {}
    bcs_DR_inhom = {}
    
    if right_sup is None:
        right_sup = lx
    
    # It is crucial to not include on_boundary for pointwise boundary condition
    ll = left_sup - left_sup_w / 2.
    assert (0.<=ll)
    if left_sup_w==0. or x_fix=='left':
        def left_bot_point(x, on_boundary):
            return df.near(x[0], ll, tol) and df.near(x[1], 0., tol)
    if left_sup_w==0.:
        if x_fix=='left': # already defined left_bot_point
            left_bot = left_bot_point
        else:
            def left_bot(x, on_boundary):
                return df.near(x[0], left_sup, tol) and df.near(x[1], 0., tol)
        mm = boundary_condition_pointwise
    else:
        def left_bot(x, on_boundary):
            return on_boundary and df.between(x[0], (ll - tol, ll + left_sup_w + tol)) and df.near(x[1], 0., tol)
        mm = boundary_condition
        
    if x_fix=='left':
        bc_left_x, bc_left_x_dofs = boundary_condition_pointwise(i_u.sub(0), df.Constant(0.0), left_bot_point) # fix in x-direction at (x[0]=0, x[1]=0)
        if len(bc_left_x_dofs) == 0:
            raise ValueError('No DOFs were found for the left-side support. You might need to redefine the mesh.')
        bcs_DR.update({'left_x': {'bc': bc_left_x, 'bc_dofs': bc_left_x_dofs}})
    bc_left_y, bc_left_y_dofs = mm(i_u.sub(1), df.Constant(0.0), left_bot)
    if len(bc_left_y_dofs) == 0:
        raise ValueError('No DOFs were found for the left-side support. You might need to redefine the mesh.')
    bcs_DR.update({'left_y': {'bc': bc_left_y, 'bc_dofs': bc_left_y_dofs}})
    
    ## on the middle
    if u_expr is not None:
        bc_middle_top, bc_middle_top_dofs, middle_top = bc_on_middle_3point_bending(x_from, x_to, ly, i_u, u_expr, tol=tol)
        bcs_DR_inhom.update({'middle_top_y': {'bc': bc_middle_top, 'bc_dofs': bc_middle_top_dofs}})
    
    
    rr = right_sup + right_sup_w / 2.
    assert (rr<=lx)
    if right_sup_w==0. or x_fix!='left':
        def right_bot_point(x, on_boundary):
            return df.near(x[0], rr, tol) and df.near(x[1], 0., tol)
    if right_sup_w==0.:
        if x_fix!='left': # already defined right_bot_point
            right_bot = right_bot_point
        else:
            def right_bot(x, on_boundary):
                return df.near(x[0], right_sup, tol) and df.near(x[1], 0., tol)
        mm = boundary_condition_pointwise
    else:
        def right_bot(x, on_boundary):
            return on_boundary and df.between(x[0], (rr - right_sup_w - tol, rr + tol)) and df.near(x[1], 0., tol)
        mm = boundary_condition
    
    if x_fix!='left': # means on right
        bc_right_x, bc_right_x_dofs = boundary_condition_pointwise(i_u.sub(0), df.Constant(0.0), right_bot_point) # fix in x-direction at (x[0]=0, x[1]=0)
        if len(bc_right_x_dofs) == 0:
            raise ValueError('No DOFs were found for the right-side support. You might need to redefine the mesh.')
        bcs_DR.update({'right_x': {'bc': bc_right_x, 'bc_dofs': bc_right_x_dofs}})
    bc_right_y, bc_right_y_dofs = mm(i_u.sub(1), df.Constant(0.0), right_bot)
    if len(bc_right_y_dofs) == 0:
        raise ValueError('No DOFs were found for the right-side support. You might need to redefine the mesh.')
    bcs_DR.update({'right_y': {'bc': bc_right_y, 'bc_dofs': bc_right_y_dofs}})
    
    ## optional (corresponding dolfin measure)
    # mf = df.MeshFunction('size_t', mesh, 1) # dim=1 implies on edges/lines --- as of the geometry of the boundary domain
    # bc_middle_top.user_sub_domain().mark(mf, 1)
    # ds = df.Measure('ds', domain = mesh, subdomain_data = mf)
    # ds_middle_top = ds(1) # "1" goes to "middle_top"
    
    return bcs_DR, bcs_DR_inhom

def ds_on_rectangle_mesh(mesh, x_from, x_to, y_from, y_to):
    tol = mesh.rmin() / 1000.
    def middle_top(x, on_boundary):
        return on_boundary and df.between(x[0], (x_from - tol, x_to + tol)) \
                and df.between(x[1], (y_from - tol, y_to + tol))
    dom = df.AutoSubDomain(middle_top)
    mf = df.MeshFunction('size_t', mesh, 1) # 1 goes to edges
    mf.set_all(0)
    dom.mark(mf, 1)
    ds = df.Measure('ds', domain=mesh, subdomain_data=mf)
    return ds(1)

def ds_at_end_point_of_interval_mesh(mesh, x_end):
    tol = mesh.rmin() / 1000.
    def end_point(x):
        return df.near(x[0], x_end, tol)
    dom = df.AutoSubDomain(end_point)
    mf = df.MeshFunction('size_t', mesh, 0) # 0 goes to vertices
    mf.set_all(0)
    dom.mark(mf, 1)
    ds = df.Measure('ds', domain=mesh, subdomain_data=mf)
    return ds(1)

def load_and_bcs_on_cantileverBeam2d(mesh, lx, ly, i_u, u_expr):
    """
    mesh: a rectangle mesh of lx*ly
    i_u: function_space of the full displacement field
    u_expr: an expression for the displacement load
    
    For a cantilever structure ready for a modelling (e.g. to be given a material type in the future)
    , returns:
        bcs_DR, bcs_DR_dofs, bcs_DR_inhom, bcs_DR_inhom_dofs, [time_depending_expressions]
    """
    tol = mesh.rmin() / 1000.
    assert(mesh.geometric_dimension()==2)
    assert(i_u.num_sub_spaces()==2)
    bcs_DR = []
    bcs_DR_dofs = []
    bcs_DR_inhom = []
    bcs_DR_inhom_dofs = []
    
    def left_edge(x, on_boundary):
        return on_boundary and df.near(x[0], 0., tol)
    bc_left_x, bc_left_x_dofs = boundary_condition(i_u.sub(0), df.Constant(0.0), left_edge) # fix in x-direction at (x[0]=0)
    bcs_DR.append(bc_left_x)
    bcs_DR_dofs.extend(bc_left_x_dofs)
    ## fully clamped
    bc_left_y, bc_left_y_dofs = boundary_condition(i_u.sub(1), df.Constant(0.0), left_edge)
    ## free in y-direction except one node
    # def left_bot(x, on_boundary):
    #     return df.near(x[0], 0., tol) and df.near(x[1], 0., tol)
    # bc_left_y, bc_left_y_dofs = boundary_condition_pointwise(i_u.sub(1), df.Constant(0.0), left_bot)
    bcs_DR.append(bc_left_y)
    bcs_DR_dofs.extend(bc_left_y_dofs)
    
    ## load on the right-top node
    def right_top(x, on_boundary):
        return df.near(x[0], lx, tol) and df.near(x[1], ly, tol)
    bc_right, bc_right_dofs = boundary_condition_pointwise(i_u.sub(1), u_expr, right_top)
    bcs_DR_inhom.append(bc_right)
    bcs_DR_inhom_dofs.extend(bc_right_dofs)
    
    return bcs_DR, bcs_DR_dofs, bcs_DR_inhom, bcs_DR_inhom_dofs

def load_and_bcs_on_cantileverBeam3d(mesh, lx, ly, lz, i_u, u_expr):
    """
    Similar to "load_and_bcs_on_cantileverBeam2d" but for 3-D case.
    The cross section of beam is in y-z plane. In a 2-D view of beam, Z-direction is out of plane.
    """
    tol = mesh.rmin() / 1000.
    assert(mesh.geometric_dimension()==3)
    assert(i_u.num_sub_spaces()==3)
    bcs_DR = []
    bcs_DR_dofs = []
    bcs_DR_inhom = []
    bcs_DR_inhom_dofs = []
    
    ## fix x-y DOFs at the left cross section
    def left_section(x, on_boundary):
        return on_boundary and df.near(x[0], 0., tol)
    bc_left_x, bc_left_x_dofs = boundary_condition(i_u.sub(0), df.Constant(0.0), left_section)
    bcs_DR.append(bc_left_x)
    bcs_DR_dofs.extend(bc_left_x_dofs)
    bc_left_y, bc_left_y_dofs = boundary_condition(i_u.sub(1), df.Constant(0.0), left_section)
    bcs_DR.append(bc_left_y)
    bcs_DR_dofs.extend(bc_left_y_dofs)
    ## fix z DOFs at two nodes at the left cross section (the nodes must have the same z-coordinate)
    def left_2nodes(x):
        return (df.near(x[0], 0, tol) and df.near(x[1], 0, tol) and df.near(x[2], 0, tol) ) \
            or (df.near(x[0], 0, tol) and df.near(x[1], ly, tol) and df.near(x[2], 0, tol) )
    bc_left_2nodes, bc_left_2nodes_dofs = boundary_condition_pointwise(i_u.sub(2), df.Constant(0.0), left_2nodes)
    bcs_DR.append(bc_left_2nodes)
    bcs_DR_dofs.extend(bc_left_2nodes_dofs)
    
    ## load on the right-top edge
    def right_top_edge(x):
        return df.near(x[0], lx, tol) and df.near(x[1], ly, tol)
    bc_right, bc_right_dofs = boundary_condition_pointwise(i_u.sub(1), u_expr, right_top_edge)
    bcs_DR_inhom.append(bc_right)
    bcs_DR_inhom_dofs.extend(bc_right_dofs)
    
    # ## fix one corner of right-end of the beam in Z-direction (to avoid any disp. or torsion in Z-direction)
    # def right_one_corner(x):
    #     return df.near(x[0], lx, tol) and df.near(x[1], 0.0, tol) and df.near(x[2], 0.0, tol)
    # bc_corner, bc_corner_dofs = boundary_condition_pointwise(i_u.sub(2), df.Constant(0.0), right_one_corner)
    # bcs_DR.append(bc_corner)
    # bcs_DR_dofs.extend(bc_corner_dofs)
    
    return bcs_DR, bcs_DR_dofs, bcs_DR_inhom, bcs_DR_inhom_dofs

if __name__ == "__main__":
    import sys
    sys.path.append('./problems/')
    from helper_mesh_fenics import *
    
    # Examples
    degree = 1
    lx=100
    ly=10
    lz=10
    x_from = 17*lx/40
    x_to = 23*lx/40
    u_max = ly / 10
    # loading = 'Ramp'
    loading = 1.3
    
    load_expr = df.Expression('val', val=0.0, degree=0) # a dummy expression
    
    mesh = refined_rectangle_mesh(lx, ly, x_from, x_to, 1, 0.25)
    elem_u = df.VectorElement('CG', mesh.ufl_cell(), degree=degree, dim=2)
    i_u = df.FunctionSpace(mesh, elem_u)
    bend3p_bcs = load_and_bcs_on_3point_bending(mesh, lx, ly, x_from, x_to, i_u, load_expr)
    
    mesh2 = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(lx, ly), 20, 2)
    elem2_u = df.VectorElement('CG', mesh2.ufl_cell(), degree=degree, dim=2)
    i2_u = df.FunctionSpace(mesh2, elem2_u)
    cantilever_beam2d_bcs = load_and_bcs_on_cantileverBeam2d(mesh2, lx, ly, i2_u, load_expr)
    
    mesh3 = df.BoxMesh(df.Point(0.0, 0.0, 0.0), df.Point(lx, ly, lz), 20, 2, 2)
    elem3_u = df.VectorElement('CG', mesh3.ufl_cell(), degree=degree, dim=3)
    i3_u = df.FunctionSpace(mesh3, elem3_u)
    cantilever_beam3d_bcs = load_and_bcs_on_cantileverBeam3d(mesh3, lx, ly, lz, i3_u, load_expr)
    
    mesh4 = notched_rectangle_mesh(2000, 300, 20, 30)
    mesh5 = notched_rectangle_mesh(2000, 300, 20, 30, left_sup=35, right_sup=1952, _name='notched_rectangle_2')
    
    mesh6 = refined_rectangle_mesh(lx, ly, x_from, x_to, 0.3, 2)
    def middle_part(x):
        return df.between(x[0], (x_from - 1e-14, x_to + 1e-14))
    dom = df.AutoSubDomain(middle_part)
    mesh6_2 = refine_domain_of_mesh(mesh6, dom, mesh6.hmin()/3.5)
    
    print('DONE !')