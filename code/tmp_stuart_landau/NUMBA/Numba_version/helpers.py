import os
import tqdm
import json
import torch
import resource
import matplotlib
import numpy as np
import pylab as plt
from numba import jit
import seaborn as sns
from scipy import signal
from os.path import join
from copy import copy, deepcopy
from multiprocessing import Pool
from vbi.feature_extraction import Features
from vbi.Models.St_Lan_numba import Stuart_Landau
from mpl_toolkits.axes_grid1 import make_axes_locatable


def simulator(params):
    sol = Stuart_Landau(params)
    return sol.integrate_euler()


def wrapper_visualize(params, control_params):

    _par = deepcopy(params)
    _par.update(control_params)

    data = simulator(_par)
    t = data['t']
    x = data['x']

    mosaic = """
    AAB
    """

    fs = 1/(_par['dt']*_par['record_step'])
    fig = plt.figure(constrained_layout=True, figsize=(12, 3))
    ax = fig.subplot_mosaic(mosaic)
    x_avg = np.mean(x, axis=0)
    ax['A'].plot(t, x_avg.T, label="x", alpha=0.5)
    ax['A'].set_ylabel(r"$\sum$ Real $Z$", fontsize=16)

    freq, pxx = signal.welch(x, fs=fs, nperseg=4096)
    # pxx /= np.max(pxx)
    pxx_avg = np.average(pxx, axis=0)
    ax['B'].plot(freq, pxx_avg, lw=1, alpha=0.5)
    ax['B'].set_xlabel("Frequency [Hz]", fontsize=16)
    ax['B'].set_ylabel("Power", fontsize=16)
    ax['B'].set_xlim(0, 60)
    ti = _par['t_transition']
    tf = _par['t_end']
    ax['A'].set_xlim(tf-2, tf)
    ax['A'].set_xlabel("Time [s]", fontsize=16)

    idx = np.argmax(pxx_avg)
    print(f"fmax = {freq[idx]} Hz, Pxx = {pxx_avg[idx]}")


def wrapper_features(params:dict, control_params:dict, fmin:float=6.0, fmax=13.0):

    _par = deepcopy(params)
    _par.update(control_params)

    data = simulator(_par)
    t = data['t']
    x = data['x']

    fs = 1/(_par['dt']*_par['record_step'])
    freq, pxx = signal.welch(x, fs=fs, nperseg=x.shape[1]//2)

    # select frequency band
    idx = np.where((freq >= 0.0) & (freq <= 50.0))[0]
    pxx = pxx[:, idx]
    freq = freq[idx]
    pxx_avg = np.mean(pxx, axis=0)
    ind = np.argmax(pxx_avg)

    pwr = np.median(pxx, axis=0)
    ind_m = np.argmax(pwr)

    # integrate under area between fmin and fmax
    idx = np.logical_and(freq >= fmin, freq <= fmax).tolist()
    area_avg = np.trapz(pxx_avg[idx], freq[idx])
    area_med = np.trapz(pwr[idx], freq[idx])

    # plot PSD
    # plt.plot(freq, pxx_avg, lw=1, color="k")
    # plt.xlim([0, 50])
    # plt.title(f"g={par[0]:.0f}, v={par[1]:.1f}")
    # plt.savefig("figs/fig_{:03d}.png".format(i))
    # plt.close()

    return freq[ind_m], pwr[ind_m], area_med, freq[ind], pxx_avg[ind], area_avg


def batch_run(parameters:dict, par_list:list, n_jobs=1):

    num_simulations = len(par_list)

    def update_bar(_):
        pbar.update()

    with Pool(processes=n_jobs) as pool:
        with tqdm.tqdm(total=num_simulations) as pbar:
            async_results = [pool.apply_async(wrapper_features, args=(parameters,
                                                          par_list[i]),
                                              callback=update_bar)
                             for i in range(num_simulations)]
            data = [async_result.get() for async_result in async_results]

    return data


def plot_matrix(mat,
                ax,
                extent=None,
                cmap='jet',
                aspect="auto",
                interpolation="nearest",
                xlabel="x",
                ylabel="y",
                title="",
                vmax=None,
                vmin=None):

    im = ax.imshow(mat, interpolation=interpolation,
                   cmap=cmap, extent=extent,
                   vmax=vmax, vmin=vmin,
                   aspect=aspect,
                   origin="lower")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=12)
    cbar.ax.tick_params(labelsize=14)


def spectral_power(ts, fs, method='sp', **kwargs):
    '''!
    Calculate features from the spectral power of the given BOLD signal.

    parameters
    ----------
    ts: np.ndarray [n_channels, n_samples]
        signal time series
    fs: float
        sampling frequency [Hz]
    return np.ndarray
        spectral power [n_channels, num_freq_points]

    '''

    if method == 'sp':
        f, Pxx_den = signal.periodogram(ts, fs, **kwargs)
    else:
        f, Pxx_den = signal.welch(ts, fs, **kwargs)

    return f, Pxx_den


def plot_freq_spectrum(f, Pxx_den, ax, average=False, logscale="x", **kwargs):
    ''' 
    plot frequency spectrum

    parameters
    ----------
    f: np.ndarray [n_freq_points]
    Pxx_den : np.ndarray [n_channels, n_freq_points]
        frequency spectrum
    ax: matplotlib.axes
        axis to plot on
    logscale: str
        logscale of axis (default: x)
        options: x, y, xy

    '''

    y = Pxx_den.T
    if average:
        y = np.mean(y, axis=1)

    ax.plot(f, y, **kwargs)
    if logscale == "x":
        ax.set_xscale("log")
    elif logscale == "y":
        ax.set_yscale("log")
    elif logscale == "xy":
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_ylabel('PSD [V**2/Hz]', fontsize=16)
    ax.set_xlabel("f [Hz]", fontsize=13)
    ax.tick_params(labelsize=13)
    ax.margins(x=0)


def PSD_under_area(f, pxx, opt=None):

    normalize = opt['normalize']
    fmin = opt['fmin']
    fmax = opt['fmax']

    if normalize:
        pxx = pxx/pxx.max()

    idx = np.logical_and(f >= fmin, f <= fmax).tolist()
    if len(idx) > 0:
        area = np.trapz(pxx[:, idx], f[idx], axis=1).reshape(-1)
        return area
    else:
        return [np.nan]
