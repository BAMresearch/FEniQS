from feniQS.structure.struct_slab2D import *

pth_struct_slab2d_long = CollectPaths('./feniQS/structure/struct_slab2D_long.py')
pth_struct_slab2d_long.add_script(pth_struct_slab2d)

class ParsSlab2DLong(ParsSlab2D):
    def __init__(self, **kwargs):
        if len(kwargs)==0: # Some values are set
            ## GEOMETRY
            self.lx = 10.
            self.ly = 1.
            
            ## Imperfection due to smaller cross section
            self.Wx = 0.04 * self.lx # length-x where E-modulus is smaller
            self.x_from = (self.lx - self.Wx) / 2.
            self.x_to = (self.lx + self.Wx) / 2.
            self.Wy = 0.4 * self.ly # length-x where E-modulus is smaller
            self.y_from = (self.ly - self.Wy) / 2.
            self.y_to = (self.ly + self.Wy) / 2.
            self.red_factor = 0.1

            ## MESH
            self.res_x = 100
            self.res_y = 10
            self.el_size_max = None
            self.el_size_min = None
            embedded_nodes = []
            ParsSlab2D.set_embedded_nodes(self, embedded_nodes)

            ## LOADs and BCs
            self.loading_control = 'u'
            self.loading_level = self.lx / 2000.
            
            self.loading_scales = [1]
            self.loading_case = 'ramp'
            self.loading_N = 1.0 # only relevant if case=sin or case=zigzag
            self.loading_T = 1.0
            self.loading_t_end = self.loading_T

            ## REST
            self._write_files = True
            self._plot = True
            
        else: # Get from a dictionary
            ParsBase.__init__(self, **kwargs)

class Slab2DLong(Slab2D):
    def __init__(self, pars, _path=None, _name=None):
        if _name is None:
            _name = 'slab2d_long'
        Slab2D.__init__(self, pars, _path, _name)
        def my_expression(x):
            if (self.pars.x_from<=x[0]<=self.pars.x_to) \
                and (self.pars.y_from<=x[1]<=self.pars.y_to):
                return self.pars.red_factor
            else:
                return 1.
        expr = SpatialExpressionFromPython(my_expression, dim=0)
        self.special_fenics_fields['sigma_scale'] = expr