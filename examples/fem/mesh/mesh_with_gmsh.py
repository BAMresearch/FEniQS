from feniQS.structure.helper_mesh_gmsh_meshio import *

if __name__=='__main__':
    _path = './examples/fem/mesh/gmsh_meshes/rectangle_mesh/'
    _name = 'notched_rectangle'
    lx=320
    ly=80
    embedded_nodes = []
    for x in np.linspace(20, lx-20, 20+1):
        for y in np.linspace(5, ly-5, 5+1):
            embedded_nodes.append([x, y])
    embedded_nodes = np.array(embedded_nodes)
    ff_msh = gmshAPI_notched_rectangle_mesh(lx=lx, ly=ly, l_notch=8, h_notch=18, el_size_max=(320-40)/20/2 \
                                            , left_sup=10., right_sup=310., embedded_nodes=embedded_nodes \
                                            , _path=_path, _name=_name)
    ff_xdmf = get_xdmf_mesh_by_meshio(ff_msh, geo_dim=2, path_xdmf=_path)
    
    import dolfin as df
    mesh = df.Mesh()
    with df.XDMFFile(ff_xdmf) as ff:
        ff.read(mesh)
    import matplotlib.pyplot as plt
    plt.figure()
    df.plot(mesh)
    plt.plot(embedded_nodes[:,0], embedded_nodes[:,1], linestyle='', marker='.', label='Embedded', color='red')
    plt.legend()
    plt.show()
    
    ns = mesh.coordinates()
    missing_ps = []
    for p in embedded_nodes:
        found = False
        for n in ns:
            if np.linalg.norm(n-p) < 1e-6:
                found=True
                break
        if not found:
            missing_ps.append(p)
    print(f"Missing embedded nodes:\n{missing_ps}\nIf any, they are outside of the domain.")