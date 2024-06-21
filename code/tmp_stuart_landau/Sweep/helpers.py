import os
import json
import torch
import resource
import matplotlib
import numpy as np
import pylab as plt
from numba import jit
import seaborn as sns
from copy import copy
from scipy import signal
from os.path import join
from vbi.feature_extraction import Features
from vbi.Models.St_Lan import Stuart_Landau as SL
from mpl_toolkits.axes_grid1 import make_axes_locatable


def simulation_wrapper(par,
                       subname,
                       params,
                       features,
                       opts=None
                       ):

    data_path = params['data_path']
    if not os.path.exists(join(data_path, "stats")):
        os.makedirs(join(data_path, "stats"))

    avg_address = join(data_path, "AVG")
    if (not os.path.exists(avg_address)) and params['RECORD_TS']:
        os.makedirs(avg_address)

    if torch.is_tensor(par):
        par = np.float64(par.numpy())
    else:
        par = copy(par)
    try:
        _ = len(par)
    except:
        par = [par]

    # if not os.path.exists(join(data_path, "AVG", f"x_{subname}.npz")):
    sol = SL(params)  # ! read parameters from pickle file
    data = sol.simulate(par)

    if params['RECORD_TS']:
        t_filename = join(data_path, "TS", "t.npz")
        if not os.path.exists(t_filename):
            np.savez(t_filename, t=data['t'].astype("f"))
        np.savez(join(data_path, "TS", f"x_{subname}"),
                    x=data['real'].astype("f"))

    # else:
    #     data = np.load(join(data_path, "TS", f"x_{subname}.npz"))

    F = Features(features, opts)
    stats_vec, info = F.calc_features(data['real'])

    # store the info in json file
    if (not info is None) and (not os.path.exists(join(data_path, "stats_info.json"))):
        with open(join(data_path, f"stats_info.json"), "w") as file_write:
            file_write.write(json.dumps(info, indent=4))

    return stats_vec.tolist()


def read_theta(filename, id):

    theta = torch.load(filename)
    if len(theta.shape) > 1:
        return theta[id, :]
    else:
        return theta[id]


def set_k_diogonal(A, k, value=0.0):

    assert(len(A.shape) == 2)
    n = A.shape[0]
    assert(k < n)

    for i in range(-k, k+1):
        a1 = np.diag(np.random.randint(1, 2, n - abs(i)), i)
        idx = np.where(a1)
        A[idx] = value

    return A


def array_has_nan(arr):
    array_sum = np.sum(arr)
    return np.isnan(array_sum)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


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


def plot_ts(t, y, figsize=(15, 3.5), step=1, **kwargs):
    '''
    plot time series

    Parameters
    ----------
    t : array
        time vector
    y : array
        time series [nnodes, ntime]
    '''

    N = y.shape[0]
    fig, ax = plt.subplots(1, figsize=figsize, sharex=True)
    ax.plot(t, y[:N, ::step].T, **kwargs)

    ax.set_ylabel("Real Z", fontsize=16)
    ax.set_xlabel("time [s]", fontsize=16)
    ax.margins(x=0)
    ax.tick_params(labelsize=14)

    return fig, ax


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
