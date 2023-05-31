"""
This plots the modified Von-Mises strain norm
    - for 2-D case
    - depending on epsilon_11 and given constant values for other strain components 
    - for different values of "k" parameters (as labels in the plot)
"""

from feniQS.material.fenics_mechanics import *
from feniQS.general.general import *
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid") # style must be one of: white, dark, whitegrid, darkgrid, ticks
except ModuleNotFoundError:
    print(f"\n\n\t{'-' * 70}\n\tWARNING: It is recommended to install 'seaborn' to get nicer plots.\n\t{'-' * 70}\n\n")

e_level = 2e-2
nu = 0.3
k_range = [0.5, 1, 5, 10, 20, 50]
e11_range = np.linspace(-e_level, e_level, 100)
e12 = e_level / 5
e22 = e_level / 2
# e22 = 0
constraint = 'PLANE_STRESS'
# constraint = 'PLANE_STRAIN'

fns_eps_eq = []
fig = plt.figure()
for i, k in enumerate(k_range):
    fns_eps_eq.append(ModifiedMises(nu, k))
    e_eq = []
    for e11 in e11_range:
        e = [[e11, e12] , [e12, e22]]
        e_3d = eps_3d(e, nu, constraint)
        e_eq.append(np.array(fns_eps_eq[i](e_3d)))
    
    plt.plot(e11_range, e_eq, label='k = ' + str(k))
    
plt.title('Modified equivalent von-mises strain vs. e11\n(' + str(constraint) + ')\ne22=' + str(e22) + ', e12=e21=' + str(e12))
plt.xlabel('e11')
plt.ylabel('Eps_eq')
plt.legend()
plt.show()