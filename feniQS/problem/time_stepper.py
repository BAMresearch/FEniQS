import numpy as np
from dolfin import Function as df_Function
from dolfin import info as df_info
import math

"""
This script is taken from:
    https://github.com/BAMresearch/fenics_helpers/blob/6f727c7d4040c1808fb7286d90f0b2d5918255a5/fenics_helpers/timestepping.py
, and partly adjusted to other classes/methods of this project.
"""

from feniQS.general.general import CollectPaths # only this is needed
pth_time_stepper = CollectPaths('time_stepper.py')

class TimeStepper:
    def __init__(self, solve_function, post_process_functions=None, time_adjusting_methods=[], time_adjusting_args={}, logger=None, \
                 solution_field=None, decrease_factor=0.5, increase_factor=1.5, increase_num_iter=5, \
                     _dt_min=1.0e-6, _dt_max = 0.1):
        # Both "solve_function" and "post_process_function" must have "t" as an input argument
        self.solve = solve_function
        self.pps = post_process_functions            
        if self.pps is None:
            self.pps = [lambda t, log: None] # a dummy postprocessor
        elif type(self.pps)!=list:
            self.pps = [self.pps] # to have always a list of postprocessors
        for pp in self.pps:
            if not hasattr(pp, 'close_files'):
                def do_nothing():
                    pass
                pp.close_files = do_nothing
        self.time_adjusting_methods = time_adjusting_methods ## each method has two inputs: t, self.time_adjusting_args
        self.time_adjusting_args = time_adjusting_args
        self.logger = logger
        self.ts = None
        
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor
        self.increase_num_iter = increase_num_iter
        self._dt_min = _dt_min
        self._dt_max = _dt_max
        self.solution_field = solution_field # only neccessary for calling "adaptive" solver
    
    def equidistant(self, t_end, dt=0.1):
        self.ts = np.append(np.arange(0, t_end, dt), t_end)
        for pp in self.pps:
            pp.checkpoints = list(self.ts) # define the checkpoints as an attribute of the corresponding post-processor and assign all time-steps to it
        iterations = []  # number of iterations per step
        for t in self.ts:
            i, converged = self.solve(t)
            iterations.append(i)
            if not converged:
                raise Exception(f"Solution step t = {t} did not converge.")
            for pp in self.pps:
                pp(t, self.logger)
        for pp in self.pps:
            pp.close_files()
        return iterations
    
    def dt_max(self, dt):
        self._dt_max = dt
        return self
    
    def dt_min(self, dt):
        self._dt_min = dt
        return self

    def adaptive(self, t_end, t_start=0.0, dt=None, checkpoints=[], show_bar=False):
        assert isinstance(self.solution_field, df_Function)
        if dt is None:
            dt = self._dt_max
        
        solution_backup = self.solution_field.vector().get_local() + 0.0
        t = t_start

        progress = get_progress(t_start, t_end, show_bar)

        # Checkpoints are reached exactly. So we add t_end to the checkpoints.
        if t_end not in checkpoints:
            checkpoints = np.append(np.array(checkpoints), t_end)
        for pp in self.pps:
            pp.checkpoints = list(checkpoints) # define and assign the same checkpoints as an attribute of the corresponding post-processor
            pp(t, self.logger)
        checkpoints = CheckPoints(checkpoints, t_start, t_end)
        
        dt0 = min(dt, self._dt_max)
        self.ts = []
        iterations = []  # number of iterations per step
        while t < t_end and not math.isclose(t, t_end):
            dt = checkpoints.timestep(t, dt0)
            # We keep track of two time steps. dt0 is the time step that
            # ignores the checkpoints. This is the one that is adapted upon
            # fast/no convergence. dt is smaller than dt0
            assert dt <= dt0
            # and coveres checkpoints.

            t += dt
            
            for m in self.time_adjusting_methods:
                m(t, **self.time_adjusting_args)
            
            num_iter, converged = self.solve(t)
            assert isinstance(converged, bool)
            assert type(num_iter) == int  # isinstance(False, int) is True...

            if converged:
                progress.success(t, dt, num_iter)
                solution_backup[:] = self.solution_field.vector().get_local()[:]
                for pp in self.pps:
                    pp(t, self.logger)
                
                # increase the time step for fast convergence
                if dt == dt0 and num_iter < self.increase_num_iter and dt < self._dt_max:
                    dt0 *= self.increase_factor
                    dt0 = min(dt0, self._dt_max)
                    if not show_bar:
                        df_info("Increasing time step to dt = {}.".format(dt0))
                self.ts.append(t)
                iterations.append(num_iter)

            else:
                progress.error(t, dt, num_iter)

                self.solution_field.vector().set_local(solution_backup)
                t -= dt

                dt0 *= self.decrease_factor
                if not show_bar:
                    df_info("Reduce time step to dt = {}.".format(dt0))
                if dt0 < self._dt_min:
                    df_info("Abort since dt({}) < _dt_min({})".format(dt0, self._dt_min))
                    return False
        for pp in self.pps:
            pp.close_files()
        return iterations
    
TERM_COLOR = {"red": "\033[31m", "green": "\033[32m"}

def colored(msg, color_name):
    return TERM_COLOR[color_name] + msg + "\033[m"


class ProgressInfo:
    def __init__(self, t_start, t_end):
        return

    def iteration_info(self, t, dt, iterations):
        return "at t = {:8.5f} after {:2} iteration(s) with dt = {:8.5f}.".format(
            t, iterations, dt
        )

    def success(self, t, dt, iterations):
        msg = "Convergence " + self.iteration_info(t, dt, iterations)
        df_info(colored(msg, "green"))

    def error(self, t, dt, iterations):
        msg = "No convergence " + self.iteration_info(t, dt, iterations)
        df_info(colored(msg, "red"))


class ProgressBar:
    def __init__(self, t_start, t_end):
        from tqdm import tqdm

        fmt = "{l_bar}{bar}{rate_fmt}"
        self._pbar = tqdm(total=t_end - t_start, ascii=True, bar_format=fmt)

    def success(self, t, dt, iterations):
        self._pbar.update(dt)
        self._pbar.set_description("dt = {:8.5f}".format(dt))

    def error(self, t, dt, iterations):
        return

    def __del__(self):
        self._pbar.close()


def get_progress(t_start, t_end, show_bar):
    return ProgressBar(t_start, t_end) if show_bar else ProgressInfo(t_start, t_end)


class CheckPoints:
    def __init__(self, points, t_start, t_end):
        self.points = np.sort(np.array(points))

        if self.points.max() > t_end or self.points.min() < t_start:
            raise RuntimeError("Checkpoints outside of integration range.")

    def _first_checkpoint_within(self, t0, t1):
        id_range = (self.points > t0) & (self.points < t1)
        points_within_dt = self.points[id_range]
        if points_within_dt.size != 0:
            for point_within_dt in points_within_dt:
                if not math.isclose(point_within_dt, t0):
                    return point_within_dt

    def timestep(self, t, dt):
        """
        Searches a checkpoint t_check within [t, t+dt]. Picks the one
        with the lowest time if multiple are found. 
        If there is such a point, the dt = t_check - t is returned.
        If nothing is found, the unmodified dt is returned.
        """
        t_check = self._first_checkpoint_within(t, t + dt)
        if t_check:
            return t_check - t
        return dt
