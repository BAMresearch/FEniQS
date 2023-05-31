"""
This is to plot the Interaction-Function used in the formulation of localizing gradient damage:
    Equation-16 of:
        https://onlinelibrary.wiley.com/doi/full/10.1002/nme.5364
"""

import numpy as np
import matplotlib.pyplot as plt
from feniQS.material.constitutive import *

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid") # style must be one of: white, dark, whitegrid, darkgrid, ticks
except ModuleNotFoundError:
    print(f"\n\n\t{'-' * 70}\n\tWARNING: It is recommended to install 'seaborn' to get nicer plots.\n\t{'-' * 70}\n\n")

sz = 18

def plot_interaction_function_exp():
    
    int_func = InteractionFunction_exp(R=0.005, eta=5)
    D = np.linspace(0.,1.0,100)
    f = [int_func(d) for d in D]
    
    plt.figure()
    plt.plot(D, f)
    plt.title('Interaction function (exponential)\n eta=' + str(int_func.eta) + ', R=' + str(int_func.R), fontsize=sz)
    plt.xlabel('D', fontsize=sz)
    plt.ylabel('g', fontsize=sz)
    plt.xticks(fontsize=sz); plt.yticks(fontsize=sz);
    plt.show()
    
if __name__ == "__main__":
    plot_interaction_function_exp()
