import os
import json 
import torch
import helpers
import warnings
import numpy as np
import pylab as plt
import networkx as nx
from os.path import join
from copy import deepcopy
from scipy import signal
import sbi.utils as utils
import multiprocessing as mp
from vbi.utility import brute_sample
warnings.filterwarnings("ignore")

data_path = "output"
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Load SC and Distances -------------------------------------------------------
SC = np.loadtxt(join("data/connectivity", "weights.txt"))
Dist = np.loadtxt(join("data/connectivity", "tract_lengths.txt")) / 1000.0
np.fill_diagonal(SC, 0.0)
np.fill_diagonal(Dist, 0.0)

SC = SC/np.max(SC)
SC = np.abs(SC) * 2.0
assert(np.trace(SC) == 0.0)
nn = num_nodes = SC.shape[0]

freq = 40.0
omega = 2*np.pi*freq * np.ones(nn)

# Define parameters -----------------------------------------------------------
params = {
    "G": 0.5,                      # global coupling strength
    "a": -5.0,                     # biforcatio parameter
    "dt": 1e-4,                    # time step [s]
    'sigma_r': 1e-4,               # noise strength
    'sigma_v': 1e-4,               # noise strength
    'omega': omega,                # natural frequency [Hz]
    "fix_seed": 0,
    "velocity": 6.0,

    "t_transition": 1.0,           # transition time [s]
    "t_end": 5.0,                 # end time        [s]

    "weights": SC,                 # weighted connection matrix
    "distances": Dist,             # distance matrix
    "record_step": 2,              # sampling every n step from mpr time series

    "data_path": data_path,        # output directory
}

BRUTE = True
NSTEPX = 10
NSTEPY = 9
N_JOBS = 5
NUM_ENSEMBLES = 1
NUM_SIMULATIONS = (NSTEPX * NSTEPY)

# Define prior ----------------------------------------------------------------
# try different ranges of G and Velocity for each given SC 
prior_PAR_min = [400.0, 1.0]    
prior_PAR_max = [1800.0, 30.0]

fs = 1.0/(params["dt"]*params["record_step"])

prior_dist = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_PAR_min),
    high=torch.as_tensor(prior_PAR_max))

theta = brute_sample(prior_dist,
                     NUM_SIMULATIONS,
                     NSTEPX,
                     NSTEPY,
                     NUM_ENSEMBLES)

par_list = []
for i in range(theta.shape[0]):
    par = {"G": theta[i, 0].item(), "V": theta[i, 1].item()}
    par_list.append(par)
num_simulations = len(theta)

low = np.asarray(prior_dist.base_dist.low.tolist())
high = np.asarray(prior_dist.base_dist.high.tolist())


# run simulations -------------------------------------------------------------
data = helpers.batch_run(params, par_list, n_jobs=N_JOBS)

x = theta[:, 0].numpy()
y = theta[:, 1].numpy()
F_MED = np.array([data[i][0] for i in range(num_simulations)]).reshape((NUM_ENSEMBLES, NSTEPX, NSTEPY))
P_MED = np.array([data[i][1] for i in range(num_simulations)]).reshape((NUM_ENSEMBLES, NSTEPX, NSTEPY))
AREA_MED = np.array([data[i][2] for i in range(num_simulations)]).reshape((NUM_ENSEMBLES, NSTEPX, NSTEPY))
F_MAX = np.array([data[i][3] for i in range(num_simulations)]).reshape((NUM_ENSEMBLES, NSTEPX, NSTEPY))
P_MAX = np.array([data[i][4] for i in range(num_simulations)]).reshape((NUM_ENSEMBLES, NSTEPX, NSTEPY))
AREA_MAX = np.array([data[i][5] for i in range(num_simulations)]).reshape((NUM_ENSEMBLES, NSTEPX, NSTEPY))

f_avg = np.nanmean(F_MAX, axis=0)
p_avg = np.nanmean(P_MAX, axis=0)
a_avg = np.nanmean(AREA_MAX, axis=0)

f_med = np.nanmedian(F_MED, axis=0)
p_med = np.nanmedian(P_MED, axis=0)
a_med = np.nanmedian(AREA_MED, axis=0)

f_avg.shape, p_avg.shape, a_avg.shape, f_med.shape, p_med.shape, a_med.shape

# save data -------------------------------------------------------------------
np.savez(join(data_path, "data_avg"),
         x=x,
         y=y,
         f_avg=f_avg,
         p_avg=p_avg,
         a_avg=a_avg,
         f_med=f_med,
         p_med=p_med,
         a_med=a_med)

data = np.load(join(data_path, "data_avg.npz"))
x = data['x']
y = data['y']
f_avg = data['f_avg']
p_avg = data['p_avg']
a_avg = data['a_avg']
f_med = data['f_med']
p_med = data['p_med']
a_med = data['a_med']

# plot data -------------------------------------------------------------------
fig, ax = plt.subplots(1, 6, figsize=(20, 4), sharey=True)
helpers.plot_matrix(f_med, ax[0], extent=[low[1], high[1], low[0], high[0]], xlabel="V", ylabel="G", title="F_MED")
helpers.plot_matrix(np.log10(p_med), ax[1], extent=[low[1], high[1], low[0], high[0]], xlabel="V", ylabel="G", title="P_MED")
helpers.plot_matrix(np.log10(a_med), ax[2], extent=[low[1], high[1], low[0], high[0]], xlabel="V", ylabel="G", title="AREA_MED")
helpers.plot_matrix(f_avg, ax[3], extent=[low[1], high[1], low[0], high[0]], xlabel="V", ylabel="G", title="F_MAX", vmax=50)
helpers.plot_matrix(np.log10(p_avg), ax[4], extent=[low[1], high[1], low[0], high[0]], xlabel="V", ylabel="G", title="P_MAX")
helpers.plot_matrix(np.log10(a_avg) , ax[5], extent=[low[1], high[1], low[0], high[0]], xlabel="V", ylabel="G", title="AREA_AVG")
plt.tight_layout()
plt.savefig(join(data_path, "fig_sweep_G_V.png"), dpi=300)
