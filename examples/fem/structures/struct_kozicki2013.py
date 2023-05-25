from feniQS.structure.struct_kozicki2013 import *

if __name__ == "__main__":
    pars = ParsKozicki2013()
    embedded_nodes = ParsKozicki2013.get_regular_grid_points(pars, include_edges=False)
    ParsBend3Point2D.set_embedded_nodes(pars, embedded_nodes)
    pars.resolutions['scale'] = 0.2
    pars.resolutions['res_y'] = 5
    _name = None if len(pars.resolutions['embedded_nodes'])==0 else f"kozicki2013_embedded_{len(pars.resolutions['embedded_nodes'])}"
    struct = Kozicki2013(pars, _name=_name)
    struct.yamlDump_pars()
    
    print(f"Num. of (cells, vertices) = ({struct.mesh.num_cells()}, {struct.mesh.num_vertices()}) .")
