from feniQS.structure.struct_bend3point2d_cutMiddle import *
from examples.fem.structures.struct_kozicki2013 import *

if __name__=='__main__':
    # Original structure
    pars = ParsKozicki2013()
    struct = Kozicki2013(pars)

    # Plot boundary nodes with mesh together
    bm = df.BoundaryMesh(struct.mesh, "exterior", True)
    plt.figure()
    df.plot(struct.mesh)
    plt.plot(bm.coordinates()[:,0], bm.coordinates()[:,1], color='red' \
             , linestyle='', marker='.', label='Boundary nodes')
    plt.legend()
    plt.show()
    
    # Build cutted structure
    struct_cut = Bend3Point2D_cutMiddle(parent_struct=struct, intervals_or_nodes=[[100,220], None])
    
    # Some visualization
    bm_cut = struct_cut._get_cutting_nodes()
    plt.figure()
    df.plot(struct_cut.mesh)
    plt.plot(bm_cut[:,0], bm_cut[:,1], color='red', linestyle='', marker='.', label='Cutting nodes')
    plt.legend()
    plt.show()