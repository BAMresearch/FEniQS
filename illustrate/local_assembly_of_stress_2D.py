from feniQS.structure.helper_mesh_fenics import *

shF_degree = 2 # concerning virtual displacement
integ_degree = 2 # Regarding Quadrature space that stores stress
ss_dim = 3 # For: sigma_xx, sigma_yy, sigma_xy

### MESH ###
mesh = one_cell_mesh_2D(0., 0., 1., 1., -0.3, 1.5) # A mesh with ONE single cell
df.plot(mesh)

### STRESS at GAUSS-POINTs (with arbitrary nonzero values) ###
assert ss_dim==3
elem_ss = df.VectorElement(family='Quadrature', cell=mesh.ufl_cell()\
                        , degree=integ_degree, dim=ss_dim, quad_scheme="default")
i_ss = df.FunctionSpace(mesh, elem_ss)
u_ss = df.Function(i_ss)
if integ_degree==1:
    u_ss.vector()[:] = [125, -52, 632]
elif integ_degree==2:
    u_ss.vector()[:] = [125, -52, 632, 1036, -899, 21, 95, -995, 11]
elif integ_degree==3:
    u_ss.vector()[:] = [125, -52, 632, 1036, -899, 21, 95, -995, 11] \
                     + [-1025, 0, -540, 13, -228, 63, 902, -12995, -1]
else:
    raise NotImplementedError()

### NODAL FORCEs (internal) ###
elem_u = df.VectorElement(family='CG', cell=mesh.ufl_cell() \
                         , degree=shF_degree, dim=2)
i_u = df.FunctionSpace(mesh, elem_u)
u_ = df.TestFunction(i_u)
def eps(v):
    e = df.sym(df.grad(v))
    return df.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])
metadata = {"quadrature_degree": integ_degree, "quadrature_scheme": "default"}
dxm = df.dx(domain=mesh, metadata=metadata)
R = df.inner(eps(u_), u_ss) * dxm # internal forces

### LOCAL ASSEMBLY ###
R_local = df.assemble_local(R, df.Cell(mesh, 0))

### EXTRACT Rx, Ry ###
dofs_cell = list(i_u.dofmap().cell_dofs(0))
dofs_x = i_u.sub(0).dofmap().dofs()
dofs_y = i_u.sub(1).dofmap().dofs()
ids_x = [dofs_cell.index(i) for i in dofs_x]
ids_y = [dofs_cell.index(i) for i in dofs_y]
R_x = R_local[ids_x]
R_y = R_local[ids_y]
cs = i_u.tabulate_dof_coordinates()
cs_x = cs[dofs_x, :]
cs_y = cs[dofs_y, :]
rot = 0. # rotational momentum
for f, r in zip(R_x, cs_x):
    rot += - f * r[1] # clockwise
for f, r in zip(R_y, cs_y):
    rot += f * r[0] # counter-clockwise

### CHECK STATIC BALANCE ###
    # Translational
print(f"Sum of translational forces (fx, fy) = ({sum(R_x):.1e}, {sum(R_y):.1e})")
assert abs(sum(R_x)) < 1e-10
assert abs(sum(R_y)) < 1e-10
    # Rotational
print(f"Rotational momentum = {rot:.1e}")
assert abs(rot) < 1e-10