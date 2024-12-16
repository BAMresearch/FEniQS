from feniQS.problem.post_process import *
from feniQS.material.shell_stress_resultant import *

pth_post_process_shell = CollectPaths('./feniQS/problem/post_process_shell.py')
pth_post_process_shell.add_script(pth_post_process)
pth_post_process_shell.add_script(pth_shell_stress_resultant)

class PostProcessShell(PostProcess):
    def __init__(self, fen, _name='', out_path=None, reaction_dofs=None \
                 , log_residual_vector=False, write_files=True, DG_degree=0, integ_degree=None):
        super().__init__(fen, _name, out_path, reaction_dofs, log_residual_vector, write_files)
        # NOTE: Stress Resultants are N:membrane, M:bending, Q:shear
        if self.write_files:
            self.DG_degree = adjust_DG_degree_to_strain_degree(DG_degree, self.fen)
            self.integ_degree = self.fen.integ_degree if integ_degree is None else integ_degree
            self._form_compiler_parameters = {"quadrature_degree":self.integ_degree,
                                              "quad_scheme":"default",
                                              }
            self.eqs_il = StressNormIlyushin(thickness=self.fen.thickness, coupling=True)
            
            ## DG spaces/functions (For visualization)
            elem_dg3 = df.VectorElement("DG", self.fen.mesh.ufl_cell(), degree=self.DG_degree, dim=3)
            self.i_dg3 = df.FunctionSpace(self.fen.mesh, elem_dg3)
            elem_dg2 = df.VectorElement("DG", self.fen.mesh.ufl_cell(), degree=self.DG_degree, dim=2)
            self.i_dg2 = df.FunctionSpace(self.fen.mesh, elem_dg2)
            elem_dg = df.FiniteElement("DG", self.fen.mesh.ufl_cell(), degree=self.DG_degree)
            self.i_dg = df.FunctionSpace(self.fen.mesh, elem_dg)
            self.N_dg = df.Function(self.i_dg3, name="ResultSig_N")
            self.M_dg = df.Function(self.i_dg3, name="ResultSig_M")
            self.Q_dg = df.Function(self.i_dg2, name="ResultSig_Q")
            self.N_eq_nrm_dg = df.Function(self.i_dg, name="N_eq_nrm") # suitable for comparing membrane/bending effects
            self.M_eq_nrm_dg = df.Function(self.i_dg, name="M_eq_nrm") # suitable for comparing membrane/bending effects
            self.sig_eq0_dg = df.Function(self.i_dg, name='EqSig_0')   # corresponding to s=0 (i.e. no coupling, see feniQS.material.shell_stress_resultant)
            self.sig_eq_dg = df.Function(self.i_dg, name='EqSig_s')    # corresponding to s=sign(P) (see feniQS.material.shell_stress_resultant)
            self.sig_eq_dg_1 = df.Function(self.i_dg, name='EqSig_s1') # corresponding to s=1 (see feniQS.material.shell_stress_resultant)
            self.sig_eq_dg_2 = df.Function(self.i_dg, name='EqSig_s2') # corresponding to s=-1 (see feniQS.material.shell_stress_resultant)

    def __call__(self, t, logger):
        super().__call__(t, logger)
        if self.write_files:
            
            ### Get stress resultants in proper format
            if self.fen._variational_approach.lower()=='ufl_original':
                N = df.as_vector([self.fen.N[0,0], self.fen.N[1,1], self.fen.N[0,1]])
                M = df.as_vector([self.fen.M[0,0], self.fen.M[1,1], self.fen.M[0,1]])
            elif self.fen._variational_approach.lower()=='c_matrices_ufl':
                N = self.fen.N
                M = self.fen.M
            elif self._variational_approach.lower()=='c_matrices_numpy':
                raise NotImplementedError('To be implemented ...')
            else:
                raise ValueError(f"Unrecognized _variational_approach: '{self._variational_approach}'.")
            
            ### Project stress resultants to DG space
            ff = df.project
            ff(v=N, V=self.i_dg3, function=self.N_dg \
                , form_compiler_parameters=self._form_compiler_parameters)
            ff(v=M, V=self.i_dg3, function=self.M_dg \
                , form_compiler_parameters=self._form_compiler_parameters)
            ff(v=self.fen.Q, V=self.i_dg2, function=self.Q_dg \
                , form_compiler_parameters=self._form_compiler_parameters)

            ### Compute equivalent stresses and assign to functions of DG space
            N_vec = self.N_dg.vector().get_local().reshape((-1, 3))
            M_vec = self.M_dg.vector().get_local().reshape((-1, 3))
            Q_vec = self.Q_dg.vector().get_local().reshape((-1, 2))
            Nx = N_vec[:,0]
            Ny = N_vec[:,1]
            Nxy = N_vec[:,2]
            Mx = M_vec[:,0]
            My = M_vec[:,1]
            Mxy = M_vec[:,2]
            qxz = Q_vec[:,0]
            qyz = Q_vec[:,1]
            Neq = self.eqs_il.eq_equivalent_N(Nx=Nx, Ny=Ny, Nxy=Nxy, normalize=True)
            Meq = self.eqs_il.eq_equivalent_M(Mx=Mx, My=My, Mxy=Mxy, normalize=True)
            sig_eq0, sig_eq = self.eqs_il.eq_stress_single(Nx=Nx, Ny=Ny, Nxy=Nxy \
                                                  , Mx=Mx, My=My, Mxy=Mxy \
                                                  , qxz=qxz, qyz=qyz)
            sig_eq_1, sig_eq_2 = self.eqs_il.eq_stresses_double(Nx=Nx, Ny=Ny, Nxy=Nxy \
                                                                , Mx=Mx, My=My, Mxy=Mxy \
                                                                , qxz=qxz, qyz=qyz)
            self.N_eq_nrm_dg.vector()[:] = Neq[:]
            self.M_eq_nrm_dg.vector()[:] = Meq[:]
            self.sig_eq0_dg.vector()[:] = sig_eq0[:]
            self.sig_eq_dg.vector()[:] = sig_eq[:]
            self.sig_eq_dg_1.vector()[:] = sig_eq_1[:]
            self.sig_eq_dg_2.vector()[:] = sig_eq_2[:]

            ### Write projected values to xdmf-files
            self.xdmf.write(self.N_dg, t)
            self.xdmf.write(self.M_dg, t)
            self.xdmf.write(self.Q_dg, t)
            self.xdmf.write(self.N_eq_nrm_dg, t)
            self.xdmf.write(self.M_eq_nrm_dg, t)
            self.xdmf.write(self.sig_eq0_dg, t)
            self.xdmf.write(self.sig_eq_dg, t)
            self.xdmf.write(self.sig_eq_dg_1, t)
            self.xdmf.write(self.sig_eq_dg_2, t)