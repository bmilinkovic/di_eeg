# since the simulation with numba is usually slower on jupyter notebook compare to python script
# (I don't know why), I will use the python script to generate the data and then load it in the notebook
# to train the neural network and analysis the results.

import os
import json 
import torch
import pickle
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
from sbi.analysis import pairplot
from vbi.utility import brute_sample
from vbi.Models.St_Lan_numba import Inference
from sbi.utils.user_input_checks import process_prior
warnings.filterwarnings("ignore")

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)


data_path = "output/inferencing"
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


NUM_SIMULATIONS = 5000
N_JOBS = 10

prior_PAR_min = [800.0, 2.5]
prior_PAR_max = [1700.0, 20.0]

fs = 1.0/(params["dt"]*params["record_step"])

prior_dist = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_PAR_min),
    high=torch.as_tensor(prior_PAR_max))

sol = Inference(params)
prior, _, _ = process_prior(prior_dist)
theta = prior.sample((NUM_SIMULATIONS,)) 

# storing prior samples
torch.save(theta, join(data_path, "theta.pt"))
torch.save(prior, join(data_path, "prior.pt"))

# control parameters for the simulation: list of dictionaries
par_list = []
for i in range(theta.shape[0]):
    par = {"G": theta[i, 0].item(), "V": theta[i, 1].item()}
    par_list.append(par)

# prepare simulation data for sbi
x = torch.as_tensor(helpers.batch_run(params, par_list), dtype=torch.float32)[:, :3] # only use first 3 features (try different features)

# Save features ---------------------------------------------------------------
torch.save(x, join(data_path, "data_x.pt"))
# x = torch.load(join(data_path, "data_x.pt"))

print(x.shape, x.mean(axis=0).tolist(), x.std(axis=0).tolist(), x.max(axis=0).values.tolist())


# Plot features ---------------------------------------------------------------
x_ = x.numpy()
fig, ax = plt.subplots(1,6, figsize=(18, 3))
ax[0].plot(theta[:, 0].numpy(), x_[:, 0], 'o', alpha=0.5, color='r', ms=1)
ax[1].plot(theta[:, 0].numpy(), np.log10(x_[:, 1]), 'o', alpha=0.5, color='r', ms=1)
ax[2].plot(theta[:, 0].numpy(), x_[:, 2], 'o', alpha=0.5, color='r', ms=1)
ax[3].plot(theta[:, 1].numpy(), x_[:, 0], 'o', alpha=0.5, color="b", ms=1)
ax[4].plot(theta[:, 1].numpy(), np.log10(x_[:, 1]), 'o', alpha=0.5, color="b", ms=1)
ax[5].plot(theta[:, 1].numpy(), x_[:, 2], 'o', alpha=0.5, color="b", ms=1)

for i in range(3):
    ax[i].set_xlabel("G", fontsize=12)
for i in range(3,6):    
    ax[i].set_xlabel("V", fontsize=12)
for i in range(6):
    ax[i].tick_params(labelsize=12)

[ax[i].set_title("frequency") for i in [0,3]]
[ax[i].set_title("power") for i in [1,4]]
[ax[i].set_title("area") for i in [2,5]]
plt.tight_layout()
plt.savefig(join(data_path, "features.png"), dpi=300)
plt.close()

# load simulation data
# x_ = torch.load(join(data_path, "data_x.pt"))
# theta = torch.load(join(data_path, "theta.pt"))
# print(x_.shape, theta.shape, x_[0,:])
# x = x_.clone().detach()

# train the neural network
obj = Inference(params)
posterior = obj.train(NUM_SIMULATIONS, prior, x[:, :], theta[:, :], num_threads=4)

# storing posterior
with open(join(data_path, f"posterior.pickle"), "wb") as cf:
    pickle.dump({"posterior": posterior}, cf)

# load posterior
# posterior = pickle.load(open(join(data_path, f"posterior.pickle"), "rb"))['posterior']

# select an observation point (from empirical data or here just from simulation data)
theta_obs = [1200.0, 20.0]
x_obs = helpers.wrapper_features(params, {"G": theta_obs[0], "V": theta_obs[1]})
x_obs = torch.tensor(x_obs, dtype=torch.float32).reshape(1,-1)

samples = obj.sample_posterior(x_obs[:, :3], 10_000, posterior)

torch.save(samples, join(data_path, "samples.pt"))
# samples = torch.load(join(data_path, "samples.pt"))



# Plot joint posterior --------------------------------------------------------
limits = [[i, j] for i, j in zip(prior_PAR_min, prior_PAR_max)]

fig, ax = pairplot(
    samples,
    points=[theta_obs],
    figsize=(5, 5),
    limits=limits,
    labels=["G", "V"],
    upper='kde',
    diag='kde',
    # title=f"n = {len(theta)}",
    points_colors="r",
    samples_colors="k",
    points_offdiag={'markersize': 10})

ax[0,0].tick_params(labelsize=14)
ax[1,1].tick_params(labelsize=14)
ax[0,0].margins(y=0)
ax[0,0].set_xlabel(r"$G$", fontsize=16)
ax[1,1].set_xlabel(r"$V$", fontsize=16)
ax[1,1].set_xticks([10, 20, 30])
ax[1,1].set_xticklabels([5, 10, 15]) #!
fig.savefig(join(data_path, "triangleplot.png"), dpi=600, bbox_inches='tight')
fig.savefig(join(data_path, "triangleplot.svg"), dpi=600, bbox_inches='tight')



# find peak of posterior ------------------------------------------------------
from vbi.utility import posterior_peaks
theta_peak = posterior_peaks(samples, labels=["G", "V"])
print(theta_peak)

# then we can check the posterior at the peak
_par = deepcopy(params)
_par["G"] = theta_peak[0]
_par["V"] = theta_peak[1]
data_peak = helpers.simulator(_par)

# posterior predictive check ...
# plot time series and PSD of peak and compare with observation point ...


