from feniQS.structure.helper_mesh_gmsh_meshio import extrude_triangled_mesh_by_meshio
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':

    ##### SURFACE MESH (input) #####
    ## (1) ##
    ff_mesh_surface = "./illustrate/files/disk.xdmf"
    ## (2) ##
    # ff_mesh_surface = './illustrate/files/mesh_2d.xdmf'
    # mesh_surface = df.RectangleMesh(df.Point(0.,0.), df.Point(1.,1.), 1, 1)
    # plt.figure()
    # df.plot(mesh_surface)
    # with df.XDMFFile(ff_mesh_surface) as ff:
    #     ff.write(mesh_surface)

    ##### EXTRUSION SETUP #####
    dz = lambda x: (np.array([0.,0.,0.5]), np.array([0.,0.,-0.2])); res=[5,4]
    # dz=(0.5,-0.2); res=[5,4]
    # dz=0.5; res=5
    
    ##### EXTRUSION #####
    ff_mesh_extruded = './illustrate/files/mesh_extruded.xdmf'
    extrude_triangled_mesh_by_meshio(ff_mesh_surface=ff_mesh_surface \
                                    , ff_mesh_extruded=ff_mesh_extruded \
                                    , dz=dz, res=res)
    
    ##### MESH OBJECTs #####
    mesh_surface = df.Mesh()
    with df.XDMFFile(ff_mesh_surface) as ff:
        ff.read(mesh_surface)
    mesh_extruded = df.Mesh()
    with df.XDMFFile(ff_mesh_extruded) as ff:
        ff.read(mesh_extruded)
    
    ##### MESSAGEs #####
    _msg = f"The extruded mesh has {mesh_extruded.num_vertices()} nodes"
    _msg += f" and {mesh_extruded.num_cells()} elements."
    _msg += f"\nIt is stored in:\n\t'{ff_mesh_extruded}' ."
    print(_msg)