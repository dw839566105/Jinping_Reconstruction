import numpy as np
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from DetectorConfig import shell
from plot_basis import *
import fit

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as f:
    truth = pd.DataFrame(f['truth'][:])
    recon = pd.DataFrame(f['recon'][:])

# 按照 FileNo, EventID 排序
truth = truth.sort_values(by=['FileNo', 'EventID'])
recon = recon.sort_values(by=['FileNo', 'EventID'])
 
with PdfPages(args.opt) as pp:
    # 能量
    truthE = plot_fit_fig(pp, truth['NPE'].values, "truth-NPE", "Entries", 50, 200, "NPE")
    reconE = plot_fit_fig(pp, recon['E'].values, "recon-E", "Entries", 0, 10, "MeV")

    ## z
    truthz = plot_fit_fig(pp, truth['z'].values, "truth-z", "Entries", -shell, shell, "m")
    reconz = plot_fit_fig(pp, recon['z'].values, "recon-z", "Entries", -shell, shell, "m")

    # vertex bias
    ## x bias
    x_bias_recon = plot_fit_fig(pp, recon['x'].values - truth['x'].values, "recon-x-bias", "Entries", -shell, shell, "m")
    ## y bias
    y_bias_recon = plot_fit_fig(pp, recon['y'].values - truth['y'].values, "recon-y-bias", "Entries", -shell, shell, "m")
    ## z bias
    z_bias_recon = plot_fit_fig(pp, recon['z'].values - truth['z'].values, "recon-z-bias", "Entries", -shell, shell, "m")
    ## r bias
    r_bias_recon = plot_fit_fig(pp, recon['r'].values - truth['r'].values, "recon-r-bias", "Entries", -shell, shell, "m")
    ## xy bias
    xy_bias_recon = plot_fit_fig(pp, recon['xy'].values - truth['xy'].values, "recon-xy-bias", "Entries", -shell, shell, "m")

    plot_hist2d(pp, truth['z'].values, recon['z'].values, "truth z", "recon z", 0, shell, 0, shell, "m", "m", 50)

    data = np.histogram2d(truth['z'].values, recon['z'].values, bins=[np.linspace(0.005, 0.655, 66), np.linspace(-0.005, 0.645, 66)])
    fig, ax = plt.subplots(figsize=(10, 10))
    m = ax.contourf(np.arange(0,650,10), np.arange(0,650,10), data[0].T, cmap='Greys', norm=LogNorm(), vmax=1000)
    ax.plot([0, 650], [0, 650], 'r')
    ax.set(xlim = (0, 650), ylim = (0, 650))
    ax.set_xlabel('True Z/mm', fontsize=18)
    ax.set_ylabel('Recon Z/mm', fontsize=18)
    plt.xticks(np.arange(0, 650, 200), ['%d' % i for i in np.arange(0,650,200)], fontsize=16)
    plt.yticks(np.arange(0, 650, 200), ['%d' % i for i in np.arange(0,650,200)], fontsize=16)
    pp.savefig(fig)

