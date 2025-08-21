import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from bozon_analysis.analysis_scripts.helper_library import fit_funcs
from mpl_toolkits.axes_grid1 import make_axes_locatable

peak_variables = ['rmbx', 'rmby', 'rmbz']

def main(ds):
    fill = ds['fill'].values
    key = ds['key_names'].values[0]

    fig = plt.figure(figsize = (5,4))
    ax = fig.add_subplot(111)
    result = {}

    if len(key) == 1:
        x = ds[key[0]].values
        xu = np.unique(x)
        y = np.nanmean(fill, axis = 1)
        yu = np.array([np.nanmean(y[x == i]) for i in xu])
        yu_e = np.array([np.nanstd(y[x == i])/np.sqrt(len(y[x == i])) for i in xu])
        ax.errorbar(xu, yu, yerr = yu_e, fmt = 'o')
        ax.set_xlabel(key[0])
        ax.set_ylabel('Survival')
        if key[0] in peak_variables:
            guess = [xu[np.argmax(yu)], (np.max(xu)-np.min(xu))/2, np.max(yu), 2]
            popt, _ = curve_fit(fit_funcs.generalized_gaussian, xu, yu, p0 = guess)
            xp = np.linspace(np.min(xu), np.max(xu), num = 1000)
            plt.plot(xp, fit_funcs.generalized_gaussian(xp, *popt), 'r-')
            ax.axvline(popt[0], color = 'k', ls = '--')
            result[key[0]] = popt[0]
    else:
        pass

    result_str = '\n'.join([f'{k}: {v:.4g}' for k, v in result.items()])
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        1.05, 0.5, result_str, transform=ax.transAxes, fontsize=6,
        verticalalignment='center', bbox=props
    )
    fig.tight_layout()

    return result, fig