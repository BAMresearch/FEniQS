# FEniQS
A library for simulating structural mechanics problems in FEniCS.

FEniQS contains the following main modules:
  ## structure:
  For definition of a structural mechanics experiment, including geometry, mesh, boundary conditions (BCs), and time-varying loadings. The problem module below, which will model such experiments, considers time as quasi-static, i.e. no dynamic effects are modelled.
  ## material:
  For handling constitutive laws such as elasticity, gradient damage, plasticity, etc.
  ## problem:
  Combining the two modules above for building up respective structural mechanics problems, and solving them. These can be performed for two main cases: static (including homogenization), and quasi-static (abbreviated as QS).
