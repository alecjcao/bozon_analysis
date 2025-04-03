import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from analysis_scripts.helper_library import fit_funcs
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main(ds):
    fill = ds['fill'].values
    second = ds['second'].values
    survival = second / fill
    survival[np.isinf(survival)] = np.nan
    key = ds['key_names'].values[0]

    result = {}

    if key[0] == 'No-Variation':
        mean_survival_site = np.nanmean(survival, axis = 0)
        fig = plt.figure(figsize = (5,4))
        ax = fig.add_subplot(111)
        im = ax.imshow(mean_survival_site.reshape(24, 16), 
                       vmin = max(0,np.min(mean_survival_site)), vmax = min(1,np.max(mean_survival_site)), cmap = 'Blues')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axvline(8-1/2, color = 'k', ls = '--')
        ax.axhline(12-1/2, color = 'k', ls = '--')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax = cax, orientation = 'vertical')
        cbar.set_label('Survival', rotation=270, labelpad = 10)
    elif len(key) == 1:
        fig = plt.figure(figsize = (5,4))
        ax = fig.add_subplot(111)
        x = ds[key[0]].values
        xu = np.unique(x)
        y = np.nanmean(survival, axis = 1)
        yu = np.array([np.nanmean(y[x == i]) for i in xu])
        yu_e = np.array([np.nanstd(y[x == i])/np.sqrt(len(y[x == i])) for i in xu])
        ax.errorbar(xu, yu, yerr = yu_e, fmt = 'o')
        ax.set_xlabel(key[0])
        ax.set_ylabel('Survival')
        if key[0] == 'spec3p1':
            xp = np.linspace(np.min(xu), np.max(xu), 100)
            if np.max(xu)-np.min(xu) < 0.11:
                guess = [xu[np.argmax(yu)], np.max(yu), .03, 0]
                popt, _ = curve_fit(fit_funcs.lorentzian, xu, yu, p0 = guess)
                ax.plot(xp, fit_funcs.lorentzian(xp, *popt), 'r-')
                result['c3p1'] = popt[0]
            elif np.max(xu)-np.min(xu) < 0.2:
                guess = [xu[np.argmax(yu)], .05, np.max(yu), 0.03, 0, .03, .015, 0]
                popt, _ = curve_fit(fit_funcs.triple_lorentzian, xu, yu, p0 = guess)
                ax.plot(xp, fit_funcs.triple_lorentzian(xp, *popt), 'r-')
            else:
                guess = [xu[np.argmax(yu)], .15, np.max(yu), 0.04, 0, .03, .015, 0]
                popt, _ = curve_fit(fit_funcs.triple_lorentzian, xu, yu, p0 = guess)
                ax.plot(xp, fit_funcs.triple_lorentzian(xp, *popt), 'r-')
            plt.axvline(popt[0], color = 'k', ls = '--')
    elif len(key) == 2:
        fig, ax = plt.subplots(2, 1, figsize = (5, 8))
        if (key[0] == 'scanax' or key[0] == 'scan2d') and key[1] == 'spec3p1':
            depth = ds[key[0]].values
            depthu = np.unique(depth)
            center = np.zeros_like(depthu)
            center_e = np.zeros_like(depthu)
            for j, d in enumerate(depthu):
                x = ds[key[1]].values[depth==d]
                xu = np.unique(x)
                y = np.nanmean(survival[depth==d], axis = 1)
                yu = np.array([np.nanmean(y[x == i]) for i in xu])
                yu_e = np.array([np.nanstd(y[x == i])/np.sqrt(len(y[x == i])) for i in xu])
                ax[0].errorbar(xu, yu, yerr = yu_e, fmt = 'o')
                guess = [xu[np.argmax(yu)], np.max(yu), .02, 0]
                popt, pcov = curve_fit(fit_funcs.lorentzian, xu, yu, p0 = guess)
                center[j] = popt[0]
                center_e[j] = np.sqrt(pcov[0,0])
                xp = np.linspace(np.min(xu), np.max(xu), 100)
                ax[0].plot(xp, fit_funcs.lorentzian(xp, *popt), '-', c = ax[0].lines[-1].get_color())
            ax[0].set_xlabel(key[1])
            ax[0].set_ylabel('Survival')
            ax[1].errorbar(depthu, center, center_e, fmt = 'o')
            popt, _ = curve_fit(fit_funcs.line, depthu, center, p0 = [0, center[np.argmin(depthu)]])
            ax[1].plot(depthu, fit_funcs.line(depthu, *popt), 'r-')
            ax[1].axhline(popt[1], color = 'k', ls = '--')
            ax[1].set_xlabel(key[0])
            ax[1].set_ylabel('Center (MHz)')
    result_str = '\n'.join([f'{k}: {v:.4g}' for k, v in result.items()])
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    if isinstance(ax, np.ndarray):
        axt = ax[-1]
    else:
        axt = ax
    axt.text(
        1.05, 0.5, result_str, transform=axt.transAxes, fontsize=6,
        verticalalignment='center', bbox=props
    )

    return result, fig