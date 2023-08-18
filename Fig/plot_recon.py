import numpy as np
import tables
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages


psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as recon:
    reconmcmc = pd.DataFrame(recon['ReconMCMC'][:])

energy_mcmc = []
r_mcmc = []
grouped = reconmcmc.groupby("EventID")
for eid, group_eid in grouped:
    energy_mcmc.append(np.mean(group_eid['E'].values))
    r_mcmc.append(np.mean(np.sqrt(group_eid['x'].values ** 2 + group_eid['y'].values ** 2 + group_eid['z'].values ** 2)))

with PdfPages(args.opt) as pp:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax1.hist(energy_mcmc, bins = 10)
    ax2.hist(r_mcmc, bins = 10)

    ax1.set_title('Energy Distribution (MCMC)')
    ax1.set_xlabel('Energy / MeV')
    ax1.set_ylabel('Events')
    ax2.set_title('r Distribution (MCMC)')
    ax2.set_xlabel('r / m')
    ax2.set_ylabel('Events')

    pp.savefig(fig1)
    pp.savefig(fig2)

