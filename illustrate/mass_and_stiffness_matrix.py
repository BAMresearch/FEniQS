from feniQS.problem.problem import *
from feniQS.fenics_helpers.fenics_functions import *

dep_dim = 1
res = 1
deg = 1

constraint = {1: 'UNIAXIAL', 2: 'PLANE_STRESS', 3:'3D'}[dep_dim]
mesh = df.UnitSquareMesh(res, res)

class FenConfig:
    shF_degree_u=deg
    el_family = 'CG'
mat = ElasticConstitutive(E=1., nu=0.2, constraint=constraint)
fen = FenicsElastic(mat, mesh, FenConfig, dep_dim=dep_dim)
fen.build_variational_functionals()
V = fen.get_iu()

p1 = [1., 0.]
p2 = [1., 1.]

plt.figure()
df.plot(fen.mesh)
cs = fen.mesh.coordinates()
cs_all = V.tabulate_dof_coordinates()
plt.plot(cs[:,0], cs[:,1], linestyle='', marker='.', label='mesh coordinates')
plt.plot(cs_all[:,0], cs_all[:,1], linestyle='', marker='+', label='Space coordinates', color='green')
plt.plot(p1[0], p1[1], linestyle='', marker='o', label='Point-1', color='blue')
plt.plot(p2[0], p2[1], linestyle='', marker='o', label='Point-2', color='red')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.9))
plt.show()

def compute_lumped_mass_matrix(fen):
    V = fen.get_iu()
    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    mass_form = df.dot(v, u) * df.dx(fen.mesh)
    
    M_consistent = df.assemble(mass_form)
    
    M_lumped = M_consistent * 1. # deepcopy
    U = df.Function(V)
    U.vector()[:] = 1.
    mass_action_form = df.action(mass_form, U)
    M_lumped.zero()
    M_lumped.set_diagonal(df.assemble(mass_action_form))
    
    return M_consistent.array(), M_lumped.array()

M_consistent, M_lumped = compute_lumped_mass_matrix(fen)

fen._set_K_tangential()
K_ = df.assemble(fen.K_t_form).array()

Vi = V if V.num_sub_spaces()==0 else V.sub(0)
p1_dof, p2_dof = dofs_at([p1, p2], V, Vi)

print(f"Lumped mass matrix at DOFs related to points 1 & 2:\n\t{M_lumped[p1_dof, p1_dof]:.3f}, {M_lumped[p2_dof, p2_dof]:.3f}")
print(f"Stiffness matrix at DOFs related to points 1 & 2:\n\t{K_[p1_dof, p1_dof]:.3f}, {K_[p2_dof, p2_dof]:.3f}")

