# FEniQS
A library for simulating structural mechanics problems in FEniCS.

FEniQS contains the following main modules:

**structure:**

for definition of a structural mechanics experiment, including **geometry**, **mesh**, **boundary conditions** (BCs), and _time-varying_ **loadings**. The time is _quasi-static_, i.e. no dynamic (inertia) effect is modelled (in the problem module below).

**material:**

for handling **constitutive laws** such as _elasticity_, _gradient damage_, _plasticity_, etc.

**problem:**

for establishing **structural mechanics problems** for desired structures and material laws coming from the two above modules, and **solving** the problems built up. These can be performed for two main cases: _static_ (no time-evolution) that also includes _homogenization_, and _quasi-static_ (abbreviated as QS).
