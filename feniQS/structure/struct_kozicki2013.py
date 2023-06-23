from feniQS.structure.struct_bend3point2d import *

pth_struct_kozicki2013 = CollectPaths('./feniQS/structure/pth_struct_kozicki2013.py')
pth_struct_kozicki2013.add_script(pth_struct_bend3point2d)

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
            
            ## LOADs and BCs
            self.fix_x_at = 'left'
            self.loading_control = 'u'
            self.loading_level = - 0.35
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