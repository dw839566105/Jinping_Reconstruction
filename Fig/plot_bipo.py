import numpy as np
import tables
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fit
from DetectorConfig import shell, npe
from plot_basis import *

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as f:
    BC = pd.DataFrame(f['bc'][:])
    recon = pd.DataFrame(f['recon'][:])
alpha_recon = recon[recon["particle"] == 0]
beta_recon = recon[recon["particle"] == 1]
alpha_BC = BC[BC["particle"] == 0]
beta_BC = BC[BC["particle"] == 1]

# 排序配对
alpha_BC = alpha_BC.sort_values(by=['EventID'])
beta_BC = beta_BC.sort_values(by=['EventID'])
alpha_recon = alpha_recon.sort_values(by=['EventID'])
beta_recon = beta_recon.sort_values(by=['EventID'])

with PdfPages(args.opt) as pp:
    # r 的接收率
    plot_hist(pp, recon['acceptz'].values, "acceptz", "Entries", "ratio")
    plot_hist(pp, recon['acceptr'].values, "acceptr", "Entries", "ratio")
    plot_scatter(pp, recon['E'].values, recon['acceptr'].values, "energy", "acceptr", "MeV", "ratio")
    plot_hist2d(pp, recon['E'].values, recon['acceptr'].values, "energy", "acceptr", 0.5, 5, 0, 0.7, "MeV", "ratio", 50)

    # alpha plot
    ## energy distribution
    fig, ax = plt.subplots()
    alpha_BCE = plot_fit(alpha_BC['E'].values, ax, "alpha_BC-E", "Entries", 0, 300, "NPE")
    pp.savefig(fig)
    plt.close(fig)
    fig, ax = plt.subplots()
    alpha_reconE = plot_fit(alpha_recon['E'].values, ax, "alpha_MCMC-E", "Entries", 0, 10, "MeV")
    pp.savefig(fig)
    plt.close(fig)

    ## scaleE
    fig, ax = plt.subplots()
    data = alpha_BC['E'].values[(alpha_BC['E'].values / alpha_BCE[0] > 0.5) & (alpha_BC['E'].values / alpha_BCE[0] < 1.5)]  / alpha_BCE[0]
    ax.hist(data, bins = 100, histtype='step', label="BC")
    x, popt, pcov = fit.fitdata(data, 0.5, 1.5, 1)
    ax.plot(x, fit.gauss(x, popt[0], popt[1], popt[2], popt[3]), label=f'BCfit: sigam/mu-{popt[1]/popt[0]:.3f}')
    data = alpha_recon['E'].values[(alpha_recon['E'].values / alpha_reconE[0] > 0.5) & (alpha_recon['E'].values / alpha_reconE[0] < 1.5)] / alpha_reconE[0]
    ax.hist(data, bins = 100, histtype='step', label="MCMC")
    x, popt, pcov = fit.fitdata(data, 0.5, 1.5, 1)
    ax.plot(x, fit.gauss(x, popt[0], popt[1], popt[2], popt[3]), label=f'MCMCfit: sigam/mu-{popt[1]/popt[0]:.3f}')
    ax.legend()
    ax.set_title('alpha Energy Distribution')
    ax.set_xlabel('scaled-Energy / MeV')
    ax.set_ylabel("Entries")
    pp.savefig(fig)
    plt.close(fig)

    ## vertex
    ### x
    alpha_BCx = plot_fit_fig(pp, alpha_BC['x'].values, "alpha_BC-x", "Entries", -shell, shell, "m")
    alpha_reconx = plot_fit_fig(pp, alpha_recon['x'].values, "alpha_MCMC-x", "Entries", -shell, shell, "m")
    ### y
    alpha_BCy = plot_fit_fig(pp, alpha_BC['y'].values, "alpha_BC-y", "Entries", -shell, shell, "m")
    alpha_recony = plot_fit_fig(pp, alpha_recon['y'].values, "alpha_MCMC-y", "Entries", -shell, shell, "m")
    ### z
    alpha_BCz = plot_fit_fig(pp, alpha_BC['z'].values, "alpha_BC-z", "Entries", -shell, shell, "m")
    alpha_reconz = plot_fit_fig(pp, alpha_recon['z'].values, "alpha_MCMC-z", "Entries", -shell, shell, "m")
    ### r
    alpha_BCr = plot_fit_fig(pp, alpha_BC['r'].values, "alpha_BC-r", "Entries", 0, shell, "m")
    alpha_reconr = plot_fit_fig(pp, alpha_recon['r'].values, "alpha_MCMC-r", "Entries", 0, shell, "m")
    ### r^3
    alpha_BCr3 = plot_fit_fig(pp, alpha_BC['r'].values ** 3, "alpha_BC-r3", "Entries", 0, shell ** 3, "m^3")
    alpha_reconr3 = plot_fit_fig(pp, alpha_recon['r'].values ** 3, "alpha_MCMC-r3", "Entries", 0, shell ** 3, "m^3")
    ## z-xy fistribution
    plot_zxy(pp, alpha_BC['xy'].values ** 2, alpha_BC['z'], "alpha_BC")
    plot_zxy(pp, alpha_recon['xy'].values ** 2, alpha_recon['z'], "alpha_MCMC")

    # beta plot
    ## energy distribution
    fig, ax = plt.subplots()
    beta_BCE = plot_fit(beta_BC['E'].values, ax, "beta_BC-E", "Entries", 0, 300, "NPE")
    pp.savefig(fig)
    plt.close(fig)
    fig, ax = plt.subplots()
    beta_reconE = plot_fit(beta_recon['E'].values, ax, "beta_MCMC-E", "Entries", 0, 10, "MeV")
    pp.savefig(fig)
    plt.close(fig)

    ## scaleE
    fig, ax = plt.subplots()
    data = beta_BC['E'].values[(beta_BC['E'].values / beta_BCE[0] > 0.35) & (beta_BC['E'].values / beta_BCE[0] < 1.75)]  / beta_BCE[0]
    ax.hist(data, bins = 100, histtype='step', label="BC")
    x, popt, pcov = fit.fitdata(data, 0.35, 1.75, 1)
    ax.plot(x, fit.gauss(x, popt[0], popt[1], popt[2], popt[3]), label=f'BCfit: sigam/mu-{popt[1]/popt[0]:.3f}')
    data = beta_recon['E'].values[(beta_recon['E'].values / beta_reconE[0] > 0.35) & (beta_recon['E'].values / beta_reconE[0] < 1.75)] / beta_reconE[0]
    ax.hist(data, bins = 100, histtype='step', label="MCMC")
    x, popt, pcov = fit.fitdata(data, 0.35, 1.75, 1)
    ax.plot(x, fit.gauss(x, popt[0], popt[1], popt[2], popt[3]), label=f'MCMCfit: sigam/mu-{popt[1]/popt[0]:.3f}')
    ax.legend()
    ax.set_title('beta Energy Distribution')
    ax.set_xlabel('scaled-Energy / MeV')
    ax.set_ylabel("Entries")
    pp.savefig(fig)
    plt.close(fig)

    ## vertex
    ### x
    beta_BCx = plot_fit_fig(pp, beta_BC['x'].values, "beta_BC-x", "Entries", -shell, shell, "m")
    beta_reconx = plot_fit_fig(pp, beta_recon['x'].values, "beta_MCMC-x", "Entries", -shell, shell, "m")
    ### y
    beta_BCy = plot_fit_fig(pp, beta_BC['y'].values, "beta_BC-y", "Entries", -shell, shell, "m")
    beta_recony = plot_fit_fig(pp, beta_recon['y'].values, "beta_MCMC-y", "Entries", -shell, shell, "m")
    ### z
    beta_BCz = plot_fit_fig(pp, beta_BC['z'].values, "beta_BC-z", "Entries", -shell, shell, "m")
    beta_reconz = plot_fit_fig(pp, beta_recon['z'].values, "beta_MCMC-z", "Entries", -shell, shell, "m")
    ### r
    beta_BCr = plot_fit_fig(pp, beta_BC['r'].values, "beta_BC-r", "Entries", 0, shell, "m")
    beta_reconr = plot_fit_fig(pp, beta_recon['r'].values, "beta_MCMC-r", "Entries", 0, shell, "m")
    ### r^3
    beta_BCr3 = plot_fit_fig(pp, beta_BC['r'].values ** 3, "beta_BC-r3", "Entries", 0, shell ** 3, "m^3")
    beta_reconr3 = plot_fit_fig(pp, beta_recon['r'].values ** 3, "beta_MCMC-r3", "Entries", 0, shell ** 3, "m^3")
    ## z-xy fistribution
    plot_zxy(pp, beta_BC['xy'].values ** 2, beta_BC['z'], "beta_BC")
    plot_zxy(pp, beta_recon['xy'].values ** 2, beta_recon['z'], "beta_MCMC")

    # prompt delayed
    plot_hist2d(pp, alpha_BC['E'].values / npe, beta_BC['E'].values / npe, "BC-alpha energy", "BC-beta energy", 0, 5, 0, 5, "MeV", "MeV", 50)
    plot_hist2d(pp, alpha_recon['E'].values, beta_recon['E'].values, "MCMC-alpha energy", "MCMC-beta energy", 0, 5, 0, 5, "MeV", "MeV", 50)

    # distance plot
    distance_BC = np.sqrt((alpha_BC['x'].values - beta_BC['x'].values) ** 2 + (alpha_BC['y'].values - beta_BC['y'].values) ** 2 + (alpha_BC['z'].values - beta_BC['z'].values) ** 2)
    distance_recon = np.sqrt((alpha_recon['x'].values - beta_recon['x'].values) ** 2 + (alpha_recon['y'].values - beta_recon['y'].values) ** 2 + (alpha_recon['z'].values - beta_recon['z'].values) ** 2)
    distance_BC_fit = plot_fit_fig(pp, distance_BC, "distance-BC", "Entries", 0, 0.4, "m")
    distance_recon_fit = plot_fit_fig(pp, distance_recon, "distance-recon", "Entries", 0, 0.4, "m")

