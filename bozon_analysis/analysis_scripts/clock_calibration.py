import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from bozon_analysis.analysis_scripts.helper_library import fit_funcs
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main(ds):
    fill = ds['fill'].values
    second = ds['second'].values
    survival = second / fill
    survival[np.isinf(survival)] = np.nan
    key = ds['key_names'].values[0]

    fig = plt.figure(figsize = (5,4))
    ax = fig.add_subplot(111)
    result = {}

    if key[0] == 'No-Variation':
        mean_survival_site = np.nanmean(survival, axis = 0)
        ax.plot(mean_survival_site, 'o')
        ax.set_xlabel('Site')
        ax.set_ylabel('Survival')
        ax.annotate(f'Surival = {np.mean(mean_survival_site):.4f}', (0.02, 1.02), xycoords='axes fraction')
        # im = ax.imshow(mean_survival_site.reshape(24, 16), 
        #                vmin = max(0,np.min(mean_survival_site)), vmax = min(1,np.max(mean_survival_site)), cmap = 'Blues')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.axvline(8-1/2, color = 'k', ls = '--')
        # ax.axhline(12-1/2, color = 'k', ls = '--')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # cbar = fig.colorbar(im, cax = cax, orientation = 'vertical')
        # cbar.set_label('Survival', rotation=270, labelpad = 5)
    elif len(key) == 1:
        x = ds[key[0]].values
        xu = np.unique(x)
        y = np.nanmean(survival, axis = 1)
        yu = np.array([np.nanmean(y[x == i]) for i in xu])
        yu_e = np.array([np.nanstd(y[x == i])/np.sqrt(len(y[x == i])) for i in xu])
        ax.errorbar(xu, yu, yerr = yu_e, fmt = 'o')
        ax.set_xlabel(key[0])
        ax.set_ylabel('Survival')
        ax.set_ylim(0,1)
        if key[0] == 'spec3p0':
            T = ds['pitime'].values[0] * ds['clock_calib_pipulsenum'].values[0]
            guess = [np.mean(xu), 1, 2*np.pi/(2*ds['pitime'].values[0]), T, 0]
            popt, _ = curve_fit(fit_funcs.rabi_resonance, xu, yu, p0 = guess, maxfev = 100000)
            xp = np.linspace(np.min(xu), np.max(xu), 100)
            ax.plot(xp, fit_funcs.rabi_resonance(xp, *popt), 'r-')
            ax.axvline(popt[0], color = 'k', ls = '--')
            result[key[0]] = popt[0]
        elif key[0] == 'pitime' or key[0] == 'pihalvestime':
            guess = [xu[np.argmax(yu)], -(ds['clock_calib_pipulsenum'].values[0]*np.pi/(2*xu[np.argmax(yu)]))**2, 1]
            popt, _ = curve_fit(fit_funcs.parabola, xu, yu, p0 = guess)
            xp = np.linspace(np.min(xu), np.max(xu), 100)
            ax.plot(xp, fit_funcs.parabola(xp, *popt), 'r-')
            ax.axvline(popt[0], color = 'k', ls = '--')
            result[key[0]] = popt[0]
        elif key[0] == 'spec3p0_bare':
            guess = [xu[np.argmax(yu)], -(2*np.pi*ds['ramseytime'].values[0])**2, 1]
            popt, _ = curve_fit(fit_funcs.parabola, xu, yu, p0 = guess)
            xp = np.linspace(np.min(xu), np.max(xu), 100)
            ax.plot(xp, fit_funcs.parabola(xp, *popt), 'r-')
            ax.axvline(popt[0], color = 'k', ls = '--')
            result[key[0]] = popt[0]
        elif key[0] == 'ramseytime':
            current_f = ds['spec3p0_bare'].values[0]
            guess = [current_f - 11.75, 1, 0]
            popt, _ = curve_fit(fit_funcs.cosine_nooffset, xu, yu, p0 = guess,
                                bounds = (0, (np.inf, 1, 1)))
            xp = np.linspace(np.min(xu), np.max(xu), 1000)
            ax.plot(xp, fit_funcs.cosine_nooffset(xp, *popt), 'r-')
            result['spec3p0_bare'] =  current_f - popt[0]
        elif key[0] == 'clock_calib_pipulsenum':
            guess = [0.999, 0.99]
            popt, pcov = curve_fit(fit_funcs.fidelity_fit, xu, yu, p0 = guess,
                                   bounds = (0,1))
            xp = np.linspace(np.min(xu), np.max(xu), 100)
            ax.plot(xp, fit_funcs.fidelity_fit(xp, *popt), 'r-')
            ax.annotate(f'Pulse fidelity: {popt[0]:.4f}({np.sqrt(pcov[0,0])*1e4:.0f})', (0.02, 0.02), xycoords='axes fraction')
        elif key[0] == 'probetime':
            T = ds['piquarterstime'].values[0]
            guess = [1/(2*T), np.max(yu)-np.min(yu), 50, np.pi, np.mean(yu)]
            popt, _ = curve_fit(fit_funcs.gaussian_cosine, xu, yu, p0 = guess)
            xp = np.linspace(np.min(xu), np.max(xu), 1000)
            ax.plot(xp, fit_funcs.gaussian_cosine(xp, *popt), 'r-')
            ax.annotate(f'Decay time: {4*popt[2]:.1f}ms', (0.02, 1.02), xycoords='axes fraction')
    else:
        pass

    result_str = '\n'.join([f'{k}: {v:.4f}' for k, v in result.items()])
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        1.05, 0.5, result_str, transform=ax.transAxes, fontsize=6,
        verticalalignment='center', bbox=props
    )
    fig.tight_layout()

    return result, fig