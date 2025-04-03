import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from analysis_scripts.helper_library import fit_funcs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft, fftfreq

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
        if key[0] == 'ryd_freq' or key[0] == 'ryd_det':
            T = ds['uvtime'].values[0]
            guess = [xu[np.argmax(yu)], -1, 2*np.pi/(2*T), T, 1]
            popt, _ = curve_fit(fit_funcs.rabi_resonance, xu, yu, p0 = guess)
            xp = np.linspace(np.min(xu), np.max(xu), 100)
            ax.plot(xp, fit_funcs.rabi_resonance(xp, *popt), 'r-')
            ax.axvline(popt[0], color = 'k', ls = '--')
            result[key[0]] = popt[0]
        if key[0] == 'uvtime':
            xf = fftfreq(len(yu), xu[1]-xu[0])[:len(yu)//2]
            yf = np.abs(fft(yu-np.mean(yu)))[:len(yu)//2]
            xmax = xf[np.argmax(yf)]
            guess = [0, xmax, 1, 0]
            popt, _ = curve_fit(fit_funcs.cosine, xu, yu, p0 = guess)
            xp = np.linspace(np.min(xu), np.max(xu), 100)
            ax.plot(xp, fit_funcs.cosine(xp, *popt), 'r-')
            ax.axvline(1/(2*popt[1]), color = 'k', ls = '--')
            result[key[0]] = 1/(2*popt[1])
        if key[0] == 'time407':
            guess = [(yu[0]-yu[1])/yu[0], yu[0], 0]
            popt, _ = curve_fit(fit_funcs.exponential, xu, yu, p0 = guess)
            xp = np.linspace(np.min(xu), np.max(xu), 100)
            ax.plot(xp, fit_funcs.exponential(xp, *popt), 'r-')
            ax.axvline(popt[0], color = 'k', ls = '--')
            result[key[0]] = popt[0]*5
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