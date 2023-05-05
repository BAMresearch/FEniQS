from feniQS.general.parameters import *

if __name__=='__main__':
    def write_some_pars():
        from feniQS.problem.model_time_varying import QuasiStaticSolveOptions
        from feniQS.problem.problem import get_fenicsSolverOptions
        p1 = QuasiStaticSolveOptions()
        p2 = QuasiStaticSolveOptions(solver_options=get_fenicsSolverOptions())
        ParsBase.unique_write(pars_list=[p1], pars_names=['p1'], root='./uniquely_write_pars/', subdir='p1')
        ParsBase.unique_write(pars_list=[p2], pars_names=['p2'], root='./uniquely_write_pars/')
        ParsBase.unique_write(pars_list=[p1, p2], pars_names=['p1', 'p2'], root='./uniquely_write_pars/pars_group/', subdir='p1p2')
        ParsBase.unique_write(pars_list=[p1, p2], pars_names=['p1', 'p2'], root='./uniquely_write_pars/pars_group/')
        return p1, p2
    p1, p2 = write_some_pars()
    p1, p2 = write_some_pars()