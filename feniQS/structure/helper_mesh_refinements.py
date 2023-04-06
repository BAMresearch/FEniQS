#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 2022

@author: ajafari
"""

from py_fenics.model_time_varying import *

pth_helper_mesh_refinements = CollectPaths('helper_mesh_refinements.py')
pth_helper_mesh_refinements.add_script(pth_model_time_varying)

class MeshRefinementsPars(ParsBase):
    def __init__(self, n_ref_dobl, n_ref_incr):
        self.n_ref = dict()
        self.n_ref['double'] = n_ref_dobl # By doubling mesh resolution (uniformly) at each refinement step
        self.n_ref['increment'] = n_ref_incr # By incrementally increasing mesh resolution (uniformly)

class MeshRefinementsSimulations:
    def __init__(self, model_constructor, struct_constructor, evaluated_points_getter):
        """
        model_constructor:
            A subclass of 'QuasiStaticModel' which constructs a quasi-static model.
        struct_constructor:
            A subclass of 'StructureFEniCS' which constructs a FEniCS structure with all BCs and loading features.
        evaluated_points_getter (a callable):
            inputs:
                - struct: a 'StructureFEniCS' instance
                - path: where any desired plots/files regarding evaluated_points can be written/stored.
            returns:
                - a dictionary of sets of points at which displacements are evaluated (out of solved simulations).
        """
        self.model_constructor = model_constructor
        self.struct_constructor = struct_constructor
        self.evaluated_points_getter = evaluated_points_getter
    
    def __call__(self, root_path, pars, solve_options, pars_refinements):
        """
        pars (an instance of 'ParsBase'):
            merge of all parameters as input to both 'self.struct_constructor' and 'self.model_constructor'
        solve_options:
            an instance of 'QuasiStaticSolveOptions'
        pars_refinements:
            an instance of 'MeshRefinementsPars'
        """
        assert all(isinstance(o, ParsBase) for o in [pars, solve_options, pars_refinements])
        # STRUCTURE & MODEL
        struct = self.struct_constructor(pars=pars)
        model0 = self.model_constructor(pars=pars, struct=struct, root_path=root_path)
        model0.yamlDump_pars(f"{root_path}pars0.yaml")
        yamlDump_pyObject_toDict(solve_options, f"{root_path}solve_options.yaml")
        yamlDump_pyObject_toDict(pars_refinements, f"{root_path}pars_refinements.yaml")
        self.model0_name = model0._name # as backup needed later on.
        with open(f"{root_path}model0_name.txt", 'w') as f:
            f.write(self.model0_name)
        simulation0 = MeshRefinementsSimulations._solve_one(m=model0, so=solve_options)
        # EVALUATED POINTs - used for in self.write_outputs !
        self.eval_points = self.evaluated_points_getter(model0.struct, path=root_path)
        for k, cs in self.eval_points.items():
            yamlDump_array(cs, root_path + f"evaluated_points_{k}.yaml")
        # NUMBER of REFINEMENTs
        n_ref_dobl = pars_refinements.n_ref['double']
        n_ref_incr = pars_refinements.n_ref['increment']
        # DO REFINEMENTs
        simulations_dobl = self._simulate_double_refinements(root_path=root_path, model0=model0 \
                                                             , solve_options=solve_options, n_ref=n_ref_dobl)
        simulations_incr = self._simulate_incremental_refinements(root_path=root_path, model0=model0 \
                                                                  , solve_options=solve_options, n_ref=n_ref_incr)
        return {'simulation0': simulation0, 'simulations_double': simulations_dobl, 'simulations_increment': simulations_incr}
    
    def _simulate_double_refinements(self, root_path, model0, solve_options, n_ref):
        dobl_root_path, dobl_refinements_names, dobl_refinements_paths \
            = MeshRefinementsSimulations.refinements_names_and_paths( \
                root_path=root_path, model0_name=self.model0_name, n_ref=n_ref, label='double')
        make_path(dobl_root_path)
        
        refine_callable = model0.struct.refine_mesh
        
        simulations = []
        for i in range(0, n_ref):
            refine_callable(i+1)
            _name = dobl_refinements_names[i]
            model = self.model_constructor(pars=model0.pars, struct=model0.struct, root_path=dobl_root_path, _name=_name)
            simulation = MeshRefinementsSimulations._solve_one(m=model, so=solve_options)
            simulations.append(simulation)
        refine_callable(0) # back to original mesh
        return simulations
    def _simulate_incremental_refinements(self, root_path, model0, solve_options, n_ref):
        incr_root_path, incr_refinements_names, incr_refinements_paths \
            = MeshRefinementsSimulations.refinements_names_and_paths( \
                root_path=root_path, model0_name=self.model0_name, n_ref=n_ref, label='increment')
        make_path(incr_root_path)
        
        refine_callable = model0.struct.increment_mesh_resolutions
        
        simulations = []
        for i in range(0, n_ref):
            refine_callable(increment=1)
            _name = incr_refinements_names[i]
            model = self.model_constructor(pars=model0.pars, struct=model0.struct, root_path=incr_root_path, _name=_name)
            simulation = MeshRefinementsSimulations._solve_one(m=model, so=solve_options)
            simulations.append(simulation)
        refine_callable(-n_ref) # back to original mesh
        return simulations
    
    def write_outputs(self, root_path, all_outputs, **all_pars):
        all_simulations = all_outputs
        pars = all_pars['pars']
        solve_options = all_pars['solve_options']
        pars_refinements = all_pars['pars_refinements']
        
        simulation0 = all_simulations['simulation0']
        path0 = root_path + self.model0_name + '/'
        MeshRefinementsSimulations._write_one(simulation0, path0, self.model0_name, self.eval_points)
        
        for k_ref in ['double', 'increment']:
            ref_root, ref_names, ref_paths \
                = MeshRefinementsSimulations.refinements_names_and_paths( \
                    root_path=root_path, model0_name=self.model0_name, n_ref=pars_refinements.n_ref[k_ref], label=k_ref)
            sims = all_simulations[f"simulations_{k_ref}"]
            assert len(ref_names)==len(ref_paths)==len(sims)
            for i, sim in enumerate(sims):
                MeshRefinementsSimulations._write_one(sim, ref_paths[i], ref_names[i], self.eval_points)
        return all_simulations, all_pars
    
    def read_results(self, root_path, **all_pars):
        pars = all_pars['pars']
        solve_options = all_pars['solve_options']
        pars_refinements = all_pars['pars_refinements']
        if isinstance(solve_options, dict):
            solve_options = QuasiStaticSolveOptions(**solve_options)
            all_pars['solve_options'] = solve_options
        if isinstance(pars_refinements, dict):
            pars_refinements = MeshRefinementsPars(n_ref_dobl=pars_refinements['n_ref']['double'] \
                                                   , n_ref_incr=pars_refinements['n_ref']['increment'])
            all_pars['pars_refinements'] = pars_refinements
        
        with open(f"{root_path}model0_name.txt", 'r') as f:
            model0_name = f.readlines()[0]
        
        struct0 = self.struct_constructor(pars)
        eval_points = self.evaluated_points_getter(struct0, path=root_path)
        
        ## CHECK that evaluated_points are the same as what simulation results have been written for.
        eval_points_read = dict()
        eval_points_missing = dict()
        for k, cs in eval_points.items():
            try:
                ff = f"{root_path}evaluated_points_{k}.yaml"
                eval_points_read[k] = yamlLoad_array(ff)
                if not np.linalg.norm(cs - eval_points_read[k]) < 1e-10:
                    _msg = f"The evaluated points {k} do not match the ones for which simulations have been done."
                    _msg += f"\nCheck your given callable 'evaluated_points_getter' or remove the file '{ff}' and "
                    _msg += "run the script again, so that the displacement solution will be interpolated again."
                    raise ValueError(_msg)
            except FileNotFoundError:
                eval_points_missing.update({k: cs})
                yamlDump_array(cs, root_path + f"evaluated_points_{k}.yaml")
                print(f"----- MeshRefinementsSimulations -----\nThe simulations will be rebuilt and interpolated for the evaluated points '{k}'.")
                # raise NotImplementedError(f"Not implemented for computing displacements at different points than the ones for which simulations have been done.")
        
        path0 = root_path + model0_name + '/'
        simulation0 = MeshRefinementsSimulations._read_one(path0, model0_name, solve_options.reaction_places \
                                                           , eval_points_read, eval_points_missing)
        all_simulations = {'simulation0': simulation0}
        for k_ref in ['double', 'increment']:
            all_simulations[f"simulations_{k_ref}"] = []
            ref_root, ref_names, ref_paths \
                = MeshRefinementsSimulations.refinements_names_and_paths( \
                    root_path=root_path, model0_name=model0_name, n_ref=pars_refinements.n_ref[k_ref], label=k_ref)
            for i, path in enumerate(ref_paths):
                sim = MeshRefinementsSimulations._read_one(path, ref_names[i], solve_options.reaction_places \
                                                           , eval_points_read, eval_points_missing)
                all_simulations[f"simulations_{k_ref}"].append(sim)
            
        return all_simulations, all_pars
        
    @staticmethod
    def refinements_names_and_paths(root_path, model0_name, n_ref, label):
        if len(label)>0:
            label = '_' + label
        root = root_path + model0_name + f"_refinements{label}/" # Root of all models refined by incrementing mesh size
        names = [f"refine{label}_{i+1}" for i in range(0, n_ref)] # Names of individual refined models
        paths = [f"{root}{n}/" for n in names] # Full paths of individual refined models
        return root, names, paths
    
    @staticmethod
    def _write_one(sim, path, name, eval_points):
        for k, f in sim['forces'].items():
            yamlDump_array(f, f"{path}{name}_checked_force_{k}.yaml")
        others = {k: sim[k] for k in ['full_size', 'r_ave', 'r_min', 'r_max']}
        yamlDump_pyObject_toDict(others, f"{path}{name}_other_features.yaml")
        pp0 = sim['pp0']
        us = dict()
        for k, cs in eval_points.items():
            us[k] = np.array(pp0.eval_checked_u(cs))
            yamlDump_array(us[k], f"{path}{name}_checked_u_{k}.yaml")
        sim.update({'us': us})
    
    @staticmethod
    def _read_one(path, name, reaction_places, eval_points_read, eval_points_missing={}):
        sim = dict() # a different model/simulation
        fs = dict()
        for rp in reaction_places:
            fs[rp] = yamlLoad_array(f"{path}{name}_checked_force_{rp}.yaml")
        sim['forces'] = fs
        us = dict()
        for k in eval_points_read.keys():
            us[k] = yamlLoad_array(f"{path}{name}_checked_u_{k}.yaml")
        
        """
        The following 'if' session concerns:
            re-building of the solution from xdmf files and interpolation of displacements for missing points (if any exists).
        This piece of code is very much hard-coded w.r.t. py_fenics.post_process.py script
            , where an 'xdmf file' for writing the displacements at checkpoints is stored.
        """
        if len(eval_points_missing) > 0:
            for k, cs in eval_points_missing.items():
                us[k] = []
            pars = yamlLoad_asDict(f"{path}pars.yaml")
            so = yamlLoad_asDict(f"{path}solve_options.yaml")
            checkpoints = so['checkpoints']
            if so['t_end'] not in checkpoints:
                checkpoints += [so['t_end']]
            num_checkpoints = len(checkpoints)
            mesh = df.Mesh()
            with df.XDMFFile(f"{path}{name}_mesh.xdmf") as ff:
                ff.read(mesh)
            dim = mesh.geometric_dimension()
            if dim==1:
                elem_u = df.FiniteElement(pars['el_family'], mesh.ufl_cell(), pars['shF_degree_u'])
            else:
                elem_u = df.VectorElement(pars['el_family'], mesh.ufl_cell(), pars['shF_degree_u'], dim=dim)
            i_u = df.FunctionSpace(mesh, elem_u)
            u_read = df.Function(i_u)
            
            with df.XDMFFile(f"{path}{name}_checked.xdmf") as ff:
                for ts in range(num_checkpoints):
                    ff.read_checkpoint(u_read, 'u', ts)
                    for k, cs in eval_points_missing.items():
                        u_ts = [u_read(c) for c in cs]
                        us[k].append(np.array(u_ts))
            for k, cs in eval_points_missing.items():
                us[k] = np.array(us[k])
                yamlDump_array(us[k], f"{path}{name}_checked_u_{k}.yaml")
            
        sim['us'] = us
        others = yamlLoad_asDict(f"{path}{name}_other_features.yaml")
        sim.update({k: v for k,v in others.items()})
        return sim
    
    @staticmethod
    def _solve_one(m, so):
        simulation = {'model': m}
        m.solve(so)
        simulation.update({'pp0': m.pps[0]})
        fs = dict()
        Fs = m.pps[0].eval_checked_reaction_forces() # list of lists of lists
        for ip, rp in enumerate(so.reaction_places):
            fs[rp] = np.array([sum(ff) for ff in Fs[ip]])
        simulation.update({'forces': fs})
        sz = m.fen.get_i_full().dim()
        simulation.update({'full_size': sz})
        r_ave = float(np.mean((get_element_volumes(m.struct.mesh) / np.pi) ** 0.5))
        r_min = m.struct.mesh.rmin()
        r_max = m.struct.mesh.rmax()
        simulation.update({'r_ave': r_ave})
        simulation.update({'r_min': r_min})
        simulation.update({'r_max': r_max})
        return simulation
    
    @staticmethod
    def plot_error_to_finest(simulation0, simulations, all_pars, what, key, descr, path, sz=14):
        make_path(path)
        # error_norm = np.std
        error_norm = np.linalg.norm
        
        hs = [sim['r_ave'] for sim in [simulation0] + simulations]
        All = [sim[what] for sim in [simulation0] + simulations]
        ref_at_key = All[-1][key]
        ref_at_key_norm = error_norm(ref_at_key)
        err_at_key = [a[key] - ref_at_key for a in All[:-1]]
        
        err_at_key_norm = [error_norm(e) / ref_at_key_norm for e in err_at_key]
        
        plt.figure()
        plt.plot(hs[:-1], err_at_key_norm, marker='.', linestyle='-')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Ave. mesh size', fontsize=sz)
        plt.ylabel('Error', fontsize=sz)
        plt.title(f"Error (to finest mesh): norm, relative\n{what}, {key}\n{descr}", fontsize=sz)
        plt.savefig(f"{path}/error_to_finest_{descr}_{what}_{key}.png", bbox_inches='tight', dpi=300)
        plt.show()

if __name__=='__main__':
    pass


# =============================================================================
# OLD METHOD (not used)
# =============================================================================

# def simulate_QSM_with_refined_meshes(model, solve_options, refine_callable \
#                                      , refine_iterator=None, n_refinements=3, points={}, label=''):
#     """
#     n_refinements:
#         Number of refinements that are done (except the initial model/mesh)
#         --> in total, (n_refinements + 1) model simulations are performed.
#     refine_callable:
#         A callable whose input is an integer, denoting the level of refinement.
#     refine_iterator:
#         specifies input to 'refine_callable', which is either of:
#             'None': input is an integer between '0' (for the initial mesh of the model) and 'n_refinements'.
#             a given integer: input is '0' for the initial mesh of the model and fix (given) for any refinement.
#     points:
#         The points at which the displacamenets of each solution are evaluated.
#         If None, all nodes of the initial mesh are considered.
#     """
#     assert model.__class__.__name__ == 'QSModelGDM'
#     if len(points)==1:
#         points['mesh_coordinates'] = model.struct.mesh.coordinates() # The points of initial mesh.
#     model_name = model._name
#     model_root_path = model.root_path
#     if len(label)>1:
#         label = '_' + label
#     root_path = model_root_path + model_name + f"_refinements{label}/"
#     make_path(root_path)
#     for k, cs in points.items():
#         yamlDump_array(cs, root_path + f"evaluated_points_{k}.yaml")
#     def _solve_and_pp(m, so, ps):
#         simulation = {'model': m}
#         m.solve(so)
#         simulation.update({'pp0': m.pps[0]})
#         us = dict()
#         for k, cs in ps.items():
#             us[k] = np.array(m.pps[0].eval_checked_u(cs))
#         simulation.update({'us': us})
#         fs = dict()
#         Fs = m.pps[0].eval_checked_reaction_forces() # list of lists of lists
#         for ip, rp in enumerate(so.reaction_places):
#             fs[rp] = np.array([sum(ff) for ff in Fs[ip]])
#         simulation.update({'forces': fs})
#         sz = m.fen.get_i_full().dim()
#         simulation.update({'full_size': sz})
#         r_ave = float(np.mean((get_element_volumes(m.struct.mesh) / np.pi) ** 0.5))
#         r_min = m.struct.mesh.rmin()
#         r_max = m.struct.mesh.rmax()
#         simulation.update({'r_ave': r_ave})
#         simulation.update({'r_min': r_min})
#         simulation.update({'r_max': r_max})
#         # write to yaml files
#         for k, u in us.items():
#             yamlDump_array(u, m._path + f"{m._name}_checked_u_{k}.yaml")
#         for k, f in fs.items():
#             yamlDump_array(f, m._path + f"{m._name}_checked_force_{k}.yaml")
#         md = {'full_size': sz, 'r_ave': r_ave, 'r_min': r_min, 'r_max': r_max}
#         yamlDump_pyObject_toDict(md, m._path + f"{m._name}_other_features.yaml")
#         return simulation
#     simulations = []
#     refine_callable(0)
#     model.root_path = root_path
#     model._name = f"refine{label}_{0}"
#     simulation = _solve_and_pp(model, solve_options, points)
#     simulations.append(simulation)
#     for i in range(0, n_refinements):
#         _level = i+1 if refine_iterator is None else 1
#         refine_callable(_level)
#         _name = f"refine{label}_{i+1}"
#         model = QSModelGDM(pars=model.pars, struct=model.struct, root_path=root_path, _name=_name)
#         simulation = _solve_and_pp(model, solve_options, points)
#         simulations.append(simulation)
#     return simulations
