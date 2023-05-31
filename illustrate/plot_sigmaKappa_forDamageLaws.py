import matplotlib.pyplot as plt
from feniQS.material.damage import *

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid") # style must be one of: white, dark, whitegrid, darkgrid, ticks
except ModuleNotFoundError:
    print(f"\n\n\t{'-' * 70}\n\tWARNING: It is recommended to install 'seaborn' to get nicer plots.\n\t{'-' * 70}\n\n")
    
def plot_sig_of_K(gK, K_end, vlines_at=[], vlines_labels=[], damage_law='' \
                  , E=10, sz=16, dpi=300, _format='.png'):
    ks = np.linspace(0.0, K_end, 100)
    sigs = [(1 - gK.g_eval(k)) * E * k for k in ks]
    fig, ax = plt.subplots()
    plt.plot(ks, sigs)
    for v in vlines_at:
        plt.axvline(x=v, ls='--')
    plt.axhline(y=0, ls='-', color='black')
    plt.axvline(x=0, ls='-', color='black')
    plt.xlabel('$\kappa$', fontsize=sz)
    plt.ylabel('$\sigma$', fontsize=sz)
    plt.xticks([0] + vlines_at) # to have only label at K0
    plt.yticks([0], fontsize=sz)
    if damage_law=='Exponential':
        xs = vlines_at
        ys = [max(sigs), 0.]
        plt.plot(xs, ys, color='gray', linestyle='--')
    
    ## Replace x-label with appropriate vlines_labels
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i, _l in enumerate(vlines_labels):
        labels[1+i] = _l
    ax.set_xticklabels(labels, fontsize=sz)
    
    plt.title(damage_law + ' damage law', fontsize=sz)
    plt.savefig('./' + damage_law + '_damage_law' + _format, bbox_inches='tight', dpi=dpi)
    plt.show(block=False)

if __name__ == "__main__":
    g1 = GKLocalDamagePerfect()
    plot_sig_of_K(g1, K_end=3*g1.K0, vlines_at=[g1.K0], vlines_labels=['$\kappa_0$'], damage_law='Perfect')
    
    g2 = GKLocalDamageExponential(e0=0.2, ef=0.5) # e0 and ef can be any postive (non-zero) values.
    plot_sig_of_K(g2, K_end=1.5*(g2.e0+g2.ef), vlines_at=[g2.e0, g2.e0 + g2.ef], vlines_labels=['$e_0$', '$(e_0+e_f)$'], damage_law='Exponential')
    ## In this plot, the slop at e0 meets the x-axis at e0+ef.
    
    g3 = GKLocalDamageLinear(ko=0.2, ku=0.5)
    plot_sig_of_K(g3, K_end=g3.ku, vlines_at=[g3.ko, g3.ku], vlines_labels=['$k_o$', '$k_u$'], damage_law='Linear')
    ## In this plot, the slop at ko meets the x-axis at ku.