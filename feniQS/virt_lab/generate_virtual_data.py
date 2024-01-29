from feniQS.general.general import make_path, clean_directory, os
from feniQS.problem.QSM_GDM import *

pth_generate_virt_data = CollectPaths("./feniQS/virt_lab/generate_virtual_data.py")
pth_generate_virt_data.add_script(pth_QSM_GDM)

class YamlVirtual:
    """
    The names/folders of yaml files (only the files) for storing virtual data.
    """

    ### MODEL PARAMETERs and SOLVE_OPTIONs
    prf_model_pars = "perfect_parameters"  # Target parameters of underlying model
    solve_options = (
        "solve_options"  # As regards to a quasi-static solution of the model
    )

    ### DISPLACEMENTs
    msr_us = "measured_disps"  # The (relative) folder at which the measured displacements are stored.
    # For each folder, we have a different set of measurement points (msr_points) and respective perfect displacements (prf_us)
    msr_points = "measurement_points"  # Measurement points (same for all load-steps)
    prf_us = "perfect_disps"  # Perfect measured displacements at measurement points (same for all load-steps)
    # Regarding domains over which data is computed
    full = "full"
    middle = "middle"
    # Regarding MESH associated to displacements
    mesh = "measurement_points_mesh"  # The coordinates of this mesh is exactly the corresponding "measurement_points"
    # Regarding label of displacements data (regular/matching)
    regular = "regular"
    matching = "matching"

    ### FORCEs
    prf_f = "perfect_f"  # Perfect measured forces at certain places, which should be specified by an
    # extension; e.g. forces_left, forces_middle. (same for all load-steps)
    ### OTHERs
    xdmf = "xdmf"  # name of folder where xdmf files of results are stored
    prf_others = "perfect_others"  # Any other quantities (in a dictionary)


def _get_files_us(_path, us_label, B=".yaml"):
    A = YamlVirtual
    aa = {}
    if us_label != "":
        us_label = us_label + "_"
    aa[A.msr_points] = _path + us_label + A.msr_points + B
    aa[A.prf_us] = _path + us_label + A.prf_us + B
    aa[A.mesh] = _path + us_label + A.mesh + ".xdmf"
    return aa


def get_virtual_data_path(model_name, data_ID, root_virtual_data):
    return f"{root_virtual_data}{model_name}_{data_ID}/"


class VirtualDataNames:
    def __init__(self, model_name, data_ID, reactions, domains \
                , root_virtual_data, make_paths=True):
        """
        Attribute of this class are:
            name: Model name and also the main folder (where data is stored) w.r.t. the path: root_virtual_data.
            reactions: Extensions for files containing reaction forces.
            domains:  A list of domains over which we have displacements. Each entry is a dictionary with two items:
                where: A.full, A.middle, etc.; indicating where the displacemenst are generated.
                label: by default is None, but can be e.g. 'matching', 'regular' which indicates whether
                    displacements are generated over the same mesh nodes (of the model) or on a regular grid.
        """
        self.name = model_name
        self.path = get_virtual_data_path(model_name, data_ID, root_virtual_data)
        if make_paths:
            make_path(self.path)
        A = YamlVirtual
        self.path_xdmf = self.path + A.xdmf + "/"
        if make_paths:
            make_path(self.path_xdmf)

        self.reactions = reactions
        assert all([isinstance(D, dict) for D in domains])
        assert all([("where" in D.keys()) and ("label" in D.keys()) for D in domains])
        self.domains = domains

    def get_files(self, B=".yaml"):
        """
        Returns a dictionary (files) containing full file names of yaml files that store virtual data (experiments).
        These are for example (with A=YamlVirtual):
            files[A.prf_model_pars]: Full file name of yamle file containing perfect (target) model parameters.

            files[A.solve_options]: Full file name of yaml file containing solve_options (used for solving perfect model).

            For each domain in self.domains:
                files [domain['where']] [domain['label']] [A.msr_points] &
                files [domain['where']] [domain['label']] [A.prf_us]
                    : Full file names of the yaml files contaning measurement-points and respective displacements.
                If domain['label'] is None or empty '', then the middle layer is skipped, i.e. the dictionary is 2-layer (not 3-layer).
        """
        files = {}
        A = YamlVirtual
        files[A.prf_model_pars] = self.path + A.prf_model_pars + B
        files[A.solve_options] = self.path + A.solve_options + B
        files[A.prf_others] = self.path + A.prf_others + B
        for k in self.reactions:
            files[A.prf_f + k] = self.path + A.prf_f + "_" + k + B

        for D in self.domains:
            where = D["where"]
            label = D["label"]
            label = "" if label is None else label
            _pth_D = (
                self.path + A.msr_us + "_" + where + "/"
            )  # Displacements at that domain
            make_path(_pth_D)
            if label is None or label == "":
                files[where] = _get_files_us(_pth_D, us_label="", B=B)
            else:
                if where in files.keys():
                    files[where][label] = _get_files_us(_pth_D, us_label=label, B=B)
                else:
                    files[where] = {label: _get_files_us(_pth_D, us_label=label, B=B)}
        return files

    def _raise(self):
        raise NameError(
            "The names (of data folder and reaction sensors) for the given model_name are NOT specified."
        )


def move_files_to_path(path0, path1, formats):
    make_path(path1)
    clean_directory(path1)
    import shutil

    for file in os.listdir(path0):
        if any([file.endswith(fr) for fr in formats]):
            shutil.move(os.path.join(path0, file), path1)


def sum_reaction_forces(rfs, insert_zero=False):
    bb = [0.0] if insert_zero else []
    fs = []
    for ff in rfs:
        aa = np.array(bb + [sum(f) for f in ff])
        fs.append(aa)
    return fs


def check_virtual_data_by_imposing_disps_gdm(
    model, solve_options, rfs_sum, tol_rel=1e-8, _plot=True
):
    """
    The absolute tolerance considered is:
        tol_rel * max_f
        with
        max_f = maximum absolute value of the whole 'rfs_sum'
    """
    (
        res0,
        res_reaction,
        _path_extended,
    ) = QSModelGDM.solve_by_imposing_displacements_at_free_DOFs(
        model, solve_options, path_extension="solved_by_imposing_disps/", _plot=_plot
    )
    rfs_sum_impose = np.array(sum_reaction_forces(res_reaction))
    rfs_sum = np.array(rfs_sum)
    atol_f = tol_rel * np.max(abs(np.array(rfs_sum)))
    bb = np.allclose(res0, 0.0, atol=atol_f)
    bb = bb and np.allclose(rfs_sum_impose, rfs_sum, atol=atol_f)
    if _plot:
        for i, rf in enumerate(rfs_sum):
            rf_impose = rfs_sum_impose[i]
            plt.figure()
            plt.plot(rf, marker=".", label="first solution")
            plt.plot(
                rf_impose,
                marker="o",
                fillstyle="none",
                label="by imposing\nsolved disps.",
            )
            plt.xlabel("Load-step")
            plt.ylabel("F")
            plt.title(f"Reaction forces at place {i}")
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.savefig(
                f"{_path_extended}reaction_forces_{i}.png", bbox_inches="tight", dpi=400
            )
            plt.show()

    if bb:
        print(
            f"\n------ VIRTUAL-DATA (CHECKED):\n\t\tThe imposition of the solved displacements to the model does return the same force residuals as the ones from the solution (both at free DOFs and reaction places).\n------"
        )
    else:
        print(
            f"\n------ VIRTUAL-DATA (WARNING):\n\t\tThe imposition of the solved displacements to the model does NOT return the same force residuals as the ones from the solution (either at free DOFs or at reaction places).\n------"
        )
