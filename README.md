# FEniQS
A library for simulating structural mechanics problems in FEniCS, with several main modules:
## structure:
dealing with the definition of a structural mechanics problem, including boundary conditions (BCs) and time-varying loadings,
## material:
handling constitutive laws such as elasticity, gradient damage, plasticity, etc,
## problem:
combining the two modules above, building respective structural mechanics problems, and solving them. These can be performed for two main cases: static (including homogenization), and quasi-static (abbreviated as QS).
