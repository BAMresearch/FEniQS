from feniQS.material.shell_stress_resultant import *

if __name__=='__main__':
    coupling = True
    aa = StressNormIlyushin(coupling=coupling)
    thickness = 10.55 # should not change the results printed below
    gs = [12., -51., -24., 2., -5., 0.4, 10., -5.] # generalized stresses
    rs = StressNormIlyushin.get_resultant_components(generalized_stresses=gs, thickness=thickness)
    s_eq0, s_eq = aa.eq_stress_single(*rs, thickness=thickness)
    s_eq_1, s_eq_2 = aa.eq_stresses_double(*rs, thickness=thickness)
    bb = 'with' if coupling else 'without'
    print(f"Single equivalent stress without coupling: {s_eq0:.3f} .")
    print(f"Single equivalent stress {bb} coupling: {s_eq:.3f} .")
    print(f"Double equivalent stresses {bb} coupling: {s_eq_1:.3f}, {s_eq_2:.3f} .")