import numpy as np
import dolfin as df
from feniQS.fenics_helpers.fenics_functions import subMesh_over_nodes
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid") # style must be one of: white, dark, whitegrid, darkgrid, ticks
except ModuleNotFoundError:
    print(f"\n\n\t{'-' * 70}\n\tWARNING: It is recommended to install 'seaborn' to get nicer plots.\n\t{'-' * 70}\n\n")

def main(mesh, _alone_nodes=True):
    Cs = mesh.coordinates()
    x1, x2 = min(Cs[:,0]), max(Cs[:,0])
    x_min = x1 + 0.3 * (x2-x1)
    x_max = x1 + 0.7 * (x2-x1)
    
    cs = []
    for c in Cs:
        if c[0]>x_min and c[0]<x_max:
            cs.append(c)
    cs_ids_del = [12, 15, 56, 96, 125, 166] # some random ids of nodes
    for i_d in cs_ids_del:
        del cs[i_d]
    
    if _alone_nodes:
        y1, y2 = min(Cs[:,1]), max(Cs[:,1])
        y_mean = (y1+y2)/2
        an1 = [x1 + 0.2 * (x2-x1), y_mean]
        an2 = [x1 + 0.8 * (x2-x1), y_mean]
        cs.append(an1) # an alone node
        cs.append(an2) # an alone node
    
    cs = np.array(cs)
    
    m1, alone_nodes = subMesh_over_nodes(mesh, cs)
    m1_nodes = m1.coordinates()
    
    plt.figure()
    df.plot(m1)
    plt.plot(cs[:,0], cs[:,1], linestyle='', marker='+', label='full given nodes')
    plt.plot(m1_nodes[:,0], m1_nodes[:,1], linestyle='', marker='*', label='subMesh nodes')
    if len(alone_nodes)>0:
        plt.plot(alone_nodes[:,0], alone_nodes[:,1], linestyle='', marker='.', label='alone nodes')
    plt.legend(); plt.show()
    
if __name__=="__main__":
    mesh = df.Mesh()
    with df.XDMFFile('./illustrate/notched_rectangle.xdmf') as f:
        f.read(mesh)
    main(mesh, True)
    main(mesh, False)