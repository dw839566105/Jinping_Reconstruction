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
    bc = pd.DataFrame(f['bc'][:])
    recon = pd.DataFrame(f['recon'][:])
alpha_recon = recon[recon["particle"] == 0]
beta_recon = recon[recon["particle"] == 1]
alpha_bc = bc[bc["particle"] == 0]
beta_bc = bc[bc["particle"] == 1]

# 排序配对
alpha_bc = alpha_bc.sort_values(by=['EventID'])
beta_bc = beta_bc.sort_values(by=['EventID'])
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
    alpha_bcE = plot_fit(alpha_bc['E'].values, ax, "alpha_BC-E", "Entries", 0, 300, "NPE")
    pp.savefig(fig)
    plt.close(fig)
    fig, ax = plt.subplots()
    alpha_reconE = plot_fit(alpha_recon['E'].values, ax, "alpha_recon-E", "Entries", 0, 10, "MeV")
    pp.savefig(fig)
    plt.close(fig)

    ## scaleE
    plot_fit_fig(pp, alpha_bc['E'].values / alpha_bcE[0], "alpha_BC-scaleE", "Entries", 0, 3, "MeV")
    plot_fit_fig(pp, alpha_recon['E'].values / alpha_reconE[0], "alpha_recon-scaleE", "Entries", 0, 3, "MeV")

    fig, ax = plt.subplots()
    ax.hist(alpha_bc['E'].values / alpha_bcE[0], bins = 100, range = [0, 2.5], histtype='step', label="BC")
    ax.hist(alpha_recon['E'].values / alpha_reconE[0], bins = 100, range = [0, 2.5], histtype='step', label="recon")
    ax.legend()
    ax.set_title('Energy Distribution')
    ax.set_xlabel("Energy")
    ax.set_ylabel("Entries")
    pp.savefig(fig)
    plt.close(fig)

    ## vertex
    ### x
    alpha_bcx = plot_fit_fig(pp, alpha_bc['x'].values, "alpha_BC-x", "Entries", -shell, shell, "m")
    alpha_reconx = plot_fit_fig(pp, alpha_recon['x'].values, "alpha_recon-x", "Entries", -shell, shell, "m")
    ### y
    alpha_bcy = plot_fit_fig(pp, alpha_bc['y'].values, "alpha_BC-y", "Entries", -shell, shell, "m")
    alpha_recony = plot_fit_fig(pp, alpha_recon['y'].values, "alpha_recon-y", "Entries", -shell, shell, "m")
    ### z
    alpha_bcz = plot_fit_fig(pp, alpha_bc['z'].values, "alpha_BC-z", "Entries", -shell, shell, "m")
    alpha_reconz = plot_fit_fig(pp, alpha_recon['z'].values, "alpha_recon-z", "Entries", -shell, shell, "m")
    ### r
    alpha_bcr = plot_fit_fig(pp, alpha_bc['r'].values, "alpha_BC-r", "Entries", 0, shell, "m")
    alpha_reconr = plot_fit_fig(pp, alpha_recon['r'].values, "alpha_recon-r", "Entries", 0, shell, "m")
    ### r^3
    alpha_bcr3 = plot_fit_fig(pp, alpha_bc['r'].values ** 3, "alpha_BC-r3", "Entries", 0, shell ** 3, "m^3")
    alpha_reconr3 = plot_fit_fig(pp, alpha_recon['r'].values ** 3, "alpha_recon-r3", "Entries", 0, shell ** 3, "m^3")
    ## z-xy fistribution
    plot_zxy(pp, alpha_bc['xy'].values ** 2, alpha_bc['z'], "alpha_bc")
    plot_zxy(pp, alpha_recon['xy'].values ** 2, alpha_recon['z'], "alpha_recon")

    # beta plot
    ## energy distribution
    fig, ax = plt.subplots()
    beta_bcE = plot_fit(beta_bc['E'].values, ax, "beta_BC-E", "Entries", 0, 300, "NPE")
    pp.savefig(fig)
    plt.close(fig)
    fig, ax = plt.subplots()
    beta_reconE = plot_fit(beta_recon['E'].values, ax, "beta_recon-E", "Entries", 0, 10, "MeV")
    pp.savefig(fig)
    plt.close(fig)

    ## scaleE
    plot_fit_fig(pp, beta_bc['E'].values / beta_bcE[0], "beta_BC-scaleE", "Entries", 0, 3, "MeV")
    plot_fit_fig(pp, beta_recon['E'].values / beta_reconE[0], "beta_recon-scaleE", "Entries", 0, 3, "MeV")

    fig, ax = plt.subplots()
    ax.hist(beta_bc['E'].values / beta_bcE[0], bins = 100, range = [0, 2.5], histtype='step', label="BC")
    ax.hist(beta_recon['E'].values / beta_reconE[0], bins = 100, range = [0, 2.5], histtype='step', label="recon")
    ax.legend()
    ax.set_title('Energy Distribution')
    ax.set_xlabel("Energy")
    ax.set_ylabel("Entries")
    pp.savefig(fig)
    plt.close(fig)

    ## vertex
    ### x
    beta_bcx = plot_fit_fig(pp, beta_bc['x'].values, "beta_BC-x", "Entries", -shell, shell, "m")
    beta_reconx = plot_fit_fig(pp, beta_recon['x'].values, "beta_recon-x", "Entries", -shell, shell, "m")
    ### y
    beta_bcy = plot_fit_fig(pp, beta_bc['y'].values, "beta_BC-y", "Entries", -shell, shell, "m")
    beta_recony = plot_fit_fig(pp, beta_recon['y'].values, "beta_recon-y", "Entries", -shell, shell, "m")
    ### z
    beta_bcz = plot_fit_fig(pp, beta_bc['z'].values, "beta_BC-z", "Entries", -shell, shell, "m")
    beta_reconz = plot_fit_fig(pp, beta_recon['z'].values, "beta_recon-z", "Entries", -shell, shell, "m")
    ### r
    beta_bcr = plot_fit_fig(pp, beta_bc['r'].values, "beta_BC-r", "Entries", 0, shell, "m")
    beta_reconr = plot_fit_fig(pp, beta_recon['r'].values, "beta_recon-r", "Entries", 0, shell, "m")
    ### r^3
    beta_bcr3 = plot_fit_fig(pp, beta_bc['r'].values ** 3, "beta_BC-r3", "Entries", 0, shell ** 3, "m^3")
    beta_reconr3 = plot_fit_fig(pp, beta_recon['r'].values ** 3, "beta_recon-r3", "Entries", 0, shell ** 3, "m^3")
    ## z-xy fistribution
    plot_zxy(pp, beta_bc['xy'].values ** 2, beta_bc['z'], "beta_bc")
    plot_zxy(pp, beta_recon['xy'].values ** 2, beta_recon['z'], "beta_recon")

    # prompt delayed
    plot_hist2d(pp, alpha_bc['E'].values / npe, beta_bc['E'].values / npe, "bc-alpha energy", "bc-beta energy", 0, 5, 0, 5, "MeV", "MeV", 50)
    plot_hist2d(pp, alpha_recon['E'].values, beta_recon['E'].values, "recon-alpha energy", "recon-beta energy", 0, 5, 0, 5, "MeV", "MeV", 50)

    # distance plot
    distance_bc = np.sqrt((alpha_bc['x'].values - beta_bc['x'].values) ** 2 + (alpha_bc['y'].values - beta_bc['y'].values) ** 2 + (alpha_bc['z'].values - beta_bc['z'].values) ** 2)
    distance_recon = np.sqrt((alpha_recon['x'].values - beta_recon['x'].values) ** 2 + (alpha_recon['y'].values - beta_recon['y'].values) ** 2 + (alpha_recon['z'].values - beta_recon['z'].values) ** 2)
    distance_bc_fit = plot_fit_fig(pp, distance_bc, "distance-BC", "Entries", 0, shell, "m")
    distance_recon_fit = plot_fit_fig(pp, distance_recon, "distance-recon", "Entries", 0, shell, "m")

