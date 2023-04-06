import sys
if './' not in sys.path:
    sys.path.append('./')

from feniQS.structure.struct_bend3point2d import *

pth_struct_bend3point2d_cutMiddle = CollectPaths('struct_bend3point2d_cutMiddle.py')
pth_struct_bend3point2d_cutMiddle.add_script(pth_struct_bend3point2d)

class Bend3Point2D_cutMiddle(Bend3Point2D):
    def __init__(self, parent_struct, intervals_or_nodes, load_at_middle=True \
                 , _path=None, _name=None):
        """
        intervals_or_nodes is either:
            a 2*2 shape array (or list of numbers) representing: [ [x_from, x_to] , [y_from, y_to] ]
            or:
            a list of nodes (coordinates) that should be considered in cutted structure.
        load_at_middle:
            if True: The same load at middle of parent_struct is applied to the cutted structure being built.
        """
        if not isinstance(parent_struct, Bend3Point2D):
            raise ValueError("A full structure of type 'Bend3Point2D' should be given to be cutted.")
        if _name is None:
            _name = parent_struct._name + '_cutMiddle'
        self.parent_struct = parent_struct
        self._set_sub_nodes(intervals_or_nodes)
        self.load_at_middle = load_at_middle
        pars_dict = {'parent_struct_path': parent_struct._path} # refering to parent structure
        pars = ParsBase(parent_struct.pars, **pars_dict) # Inherit all parameters of parent_struct
        StructureFEniCS.__init__(self, pars, _path, _name) # Standard constructor, calling _build_structure overwritten below.
    
    def _set_sub_nodes(self, intervals_or_nodes):
        if len(intervals_or_nodes)<=2 and len(intervals_or_nodes[0])==2: # We are given intervals.
            Cs = self.parent_struct.mesh.coordinates()
            self.sub_nodes = []
            tol = self.parent_struct.mesh.rmin() / 1000.
            x_min, x_max = intervals_or_nodes[0]
            if len(intervals_or_nodes)==1 or intervals_or_nodes[1] is None:
                y_min, y_max = 0., self.parent_struct.pars.ly # 'self.pars' is NOT yet assigned, thus, we use 'self.parent_struct.pars'.
            else:
                y_min, y_max = intervals_or_nodes[1]
            for c in Cs:
                if c[0]>=x_min-tol and c[0]<=x_max+tol and c[1]>=y_min-tol and c[1]<=y_max+tol:
                    self.sub_nodes.append(c)
            self.sub_nodes = np.array(self.sub_nodes)
        else: # We are given nodes
            self.sub_nodes = np.array(intervals_or_nodes)
            if self.sub_nodes.shape[0]==2: # shape is: (2 * #nodes)
                self.sub_nodes = self.sub_nodes.T # shape must become: (#nodes * 2)
        if self.sub_nodes.shape[0] <=2:
            raise ValueError("There is not enough number of nodes for cutting the structure.")
    
    def _build_structure(self):
        self.mesh, self.alone_nodes = subMesh_over_nodes(self.parent_struct.mesh, self.sub_nodes)
        ## We adjust self.pars.ly to its exact value from generated subMesh. --> Veryyy slight deviation has been observed.
        self.pars.ly = float(np.max(self.mesh.coordinates()[:,1]))
        
        self.u_y_middle, self.f_y_middle = None, None
        if self.load_at_middle:
            if self.parent_struct.u_y_middle is not None:
                self.u_y_middle = self.parent_struct.u_y_middle
            elif self.parent_struct.f_y_middle is not None:
                self.f_y_middle = self.parent_struct.f_y_middle
        
        if self.pars._plot:
            plt.figure()
            df.plot(self.mesh)
            plt.title(f"Mesh of '{self._name}'")
            plt.savefig(self._path + 'meshView.png', bbox_inches='tight', dpi=300)
            plt.show()
        if self.pars._write_files:
            write_to_xdmf(self.mesh, xdmf_name=self._name+'_mesh.xdmf', xdmf_path=self._path)
    
    def get_BCs(self, i_u):
        bcs_DR = {}
        if self.load_at_middle:
            if self.u_y_middle is not None:
                time_varying_loadings = {'y_middle': self.u_y_middle}
                bcs_DR_inhom, bcs_DR_inhom_dofs, _ = bc_on_middle_3point_bending(x_from=self.pars.x_from, x_to=self.pars.x_to, ly=self.pars.ly \
                                                                            , i_u=i_u, u_expr=self.u_y_middle, tol=self.mesh.rmin()/1000.)
                bcs_DR_inhom = {'y_middle': {'bc': bcs_DR_inhom, 'bc_dofs': bcs_DR_inhom_dofs}}
            elif self.f_y_middle is not None:
                time_varying_loadings = {'y_middle': self.f_y_middle}
                bcs_DR_inhom = {}
            else:
                raise ValueError(f"No loading is recognized from the parent structure.")
        else:
            bcs_DR_inhom, time_varying_loadings = {}, {}
        return bcs_DR, bcs_DR_inhom, time_varying_loadings
    
    def get_reaction_nodes(self, reaction_places): # overwritten
        nodes = []
        tol = self.mesh.rmin() / 1000.
        for rp in reaction_places:
            if 'middle' in rp.lower():
                ps = []
                for pp in self.mesh.coordinates():
                    if pp[0]>=self.pars.x_from-tol and pp[0]<=self.pars.x_to+tol and abs(pp[1]-self.pars.ly)<tol:
                        ps.append(pp)
            else:
                raise NameError('Reaction place is not recognized.')
            nodes.append(np.array(ps))
        return nodes # A list of lists each being nodes related to a reaction place
    
    def get_cutting_dofs(self, i_full, i_u):
        nodes = self._get_cutting_nodes()
        tol = tol=self.mesh.rmin() / 1000.
        dofs = dofs_at(nodes, i_full, i_u.sub(0), tol=tol)
        dofs.extend( dofs_at(nodes, i_full, i_u.sub(1), tol=tol) )
        return dofs

    def get_corner_dofs(self, i_full, i_u):
        nodes = self._get_corner_nodes()
        tol = tol=self.mesh.rmin() / 1000.
        dofs = dofs_at(nodes, i_full, i_u.sub(0), tol=tol)
        dofs.extend( dofs_at(nodes, i_full, i_u.sub(1), tol=tol) )
        return dofs    
    
    def _get_cutting_nodes(self):
        """
        - Cutting nodes are boundary nodes of the cutted structure (self), which do NOT belong to boundary nodes of parent structure.
        - In addition, the nodes at very left/right sides of the bottom/top edges (of cutted structure) must be in cutting nodes,
            although they do belong to the boundary nodes of the parent structure.
        
        - A general approach (not implemented here):
            Cutting nodes are boundary nodes (of cutted structure) for which NOT all of connected elements in the parent structure
                belong to the cutted structure.
            OR:
            Cutting nodes are boundary nodes (of cutted structure) for which the number of connected elements in the parent structure
                differs from that number in the cutted structure.
        """
        b_nodes_all = df.BoundaryMesh(self.parent_struct.mesh, "exterior", True).coordinates()
        b_nodes = df.BoundaryMesh(self.mesh, "exterior", True).coordinates()
        tol = self.mesh.rmin() / 1000.
        nodes = []
        for c in b_nodes:
            not_found = True
            for C in b_nodes_all:
                if np.linalg.norm(C - c) < tol:
                    not_found = False # we found the node on boundaries of the parent structure
                    break
            if not_found:
                nodes.append(c)
        corner_nodes = self._get_corner_nodes()
        nodes = np.concatenate((np.array(nodes), corner_nodes), axis=0)
        return nodes
    
    def _get_corner_nodes(self, _keys=False):
        b_nodes = df.BoundaryMesh(self.mesh, "exterior", True).coordinates()
        nodes = []
        b1 = 1e10; b2 = -1e10 # x-coordinates of very left/right nodes at bottom edge (y=0.)
        t1 = 1e10; t2 = -1e10 # x-coordinates of very left/right nodes at top edge (y=self.pars.ly)
        tol = self.mesh.rmin() / 1000.
        for c in b_nodes:
            if abs(c[1]) < tol: # bottom
                b1 = min(c[0], b1)
                b2 = max(c[0], b2)
            elif abs(c[1]-self.pars.ly) < tol: # top
                t1 = min(c[0], t1)
                t2 = max(c[0], t2)
        nodes.append([b1, 0.])
        nodes.append([b2, 0.])
        nodes.append([t1, self.pars.ly])
        nodes.append([t2, self.pars.ly])
        if _keys:
            keys_ = ['bottom_left', 'bottom_right', 'top_left', 'top_right']
            nodes = {k: np.array(nodes[ik]) for ik, k in enumerate(keys_)}
        else:
            nodes = np.array(nodes)
        return nodes
        
if __name__=='__main__':
    from problems.struct_kozicki2013 import *
    pars = ParsKozicki2013()
    struct = Kozicki2013(pars)
    struct.yamlDump_pars()
    bm = df.BoundaryMesh(struct.mesh, "exterior", True)
    plt.figure()
    df.plot(struct.mesh)
    plt.plot(bm.coordinates()[:,0], bm.coordinates()[:,1], color='red', linestyle='', marker='.', label='Boundary nodes')
    plt.legend()
    plt.show()
    
    struct_cut = Bend3Point2D_cutMiddle(parent_struct=struct, intervals_or_nodes=[[100,220], None])
    struct_cut.yamlDump_pars()
    bm_cut = struct_cut._get_cutting_nodes()
    plt.figure()
    df.plot(struct_cut.mesh)
    plt.plot(bm_cut[:,0], bm_cut[:,1], color='red', linestyle='', marker='.', label='Cutting nodes')
    plt.legend()
    plt.show()
    
    