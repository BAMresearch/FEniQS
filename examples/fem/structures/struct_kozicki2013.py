from feniQS.structure.struct_bend3point2d import *

class ParsKozicki2013(ParsBend3Point2D):
    """
    The geometry/mesh/loading for the three-point bending experiment presented in:
        https://link.springer.com/content/pdf/10.1007/s11340-013-9781-y.pdf
    """
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            ## GEOMETRY
            self.lx = 320.
            self.ly = 80.
            self.l_notch = 3. # width of notch
            self.h_notch = 8. # hight of notch
            self.left_notch = (self.lx - self.l_notch) / 2
            self.left_sup = (self.lx - 240.) / 2. + 0.
            self.right_sup = self.lx - self.left_sup
            self.right_sup_w = 0.
            # self.right_sup_w = (self.left_sup - 0.) / 1.
            self.left_sup_w = self.right_sup_w
            load_span_scale = 0.014 # of total beam length
            self.x_from = (1. - load_span_scale)  * self.lx / 2. # for loading
            self.x_to = (1. + load_span_scale)  * self.lx / 2.
            self.cmod_left = self.left_notch
            self.cmod_right = self.left_notch + self.l_notch
            
            ## MESH
            self.resolutions = {'res_y': 4,
                                'scale': 0.25,
                                'embedded_nodes': (),
                                'refinement_level': 0,
                                'el_size_max': None}
                # scale (regarding mesh size) is total and towards the center with respect to the left_right sides
            
            ## LOADs and BCs
            self.fix_x_at = 'left'
            
            self.loading_control = 'u'
            self.loading_level = - 0.35
            # self.loading_level = - 0.10 # Only elastic
            
            # self.loading_control = 'f'
            # self.loading_level = -587.8071109591015 / (load_span_scale * self.lx) # Only elastic
            
            self.loading_scales = (1,)
            self.loading_case = 'ramp'
            self.loading_N = 1.0 # only relevant if case=sin or case=zigzag
            self.loading_T = 1.0
            self.loading_t_end = self.loading_T
            
            ## REST
            self._write_files = True
            self._plot = True
            
        else: # Get from a dictionary
            ParsBend3Point2D.__init__(self, **kwargs)

class Kozicki2013(Bend3Point2D):
    """
    The model of:
        https://link.springer.com/content/pdf/10.1007/s11340-013-9781-y.pdf
    """
    def __init__(self, pars, _path=None, _name=None):
        if _name is None:
            _name = 'kozicki2013'
        Bend3Point2D.__init__(self, pars=pars, _path=_path, _name=_name)

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
