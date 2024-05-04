import numpy as np
import tables
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fit
from DetectorConfig import shell
from plot_basis import plot_fit_fig

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
    # alpha plot
    ## energy distribution
    alpha_bcE = plot_fit_fig(pp, alpha_bc['E'].values, "alpha_BC-E", 0, 10, "MeV")
    alpha_reconE = plot_fit_fig(pp, alpha_recon['E'].values, "alpha_recon-E", 0, 10, "MeV")
    ## vertex
    ### x
    alpha_bcx = plot_fit_fig(pp, alpha_bc['x'].values, "alpha_BC-x", -shell, shell, "m")
    alpha_reconx = plot_fit_fig(pp, alpha_recon['x'].values, "alpha_recon-x", -shell, shell, "m")
    ### y
    alpha_bcy = plot_fit_fig(pp, alpha_bc['y'].values, "alpha_BC-y", -shell, shell, "m")
    alpha_recony = plot_fit_fig(pp, alpha_recon['y'].values, "alpha_recon-y", -shell, shell, "m")
    ### z
    alpha_bcz = plot_fit_fig(pp, alpha_bc['z'].values, "alpha_BC-z", -shell, shell, "m")
    alpha_reconz = plot_fit_fig(pp, alpha_recon['z'].values, "alpha_recon-z", -shell, shell, "m")
    ### r
    alpha_bcr = plot_fit_fig(pp, alpha_bc['r'].values, "alpha_BC-r", 0, shell, "m")
    alpha_reconr = plot_fit_fig(pp, alpha_recon['r'].values, "alpha_recon-r", 0, shell, "m")
    ### r^3
    alpha_bcr3 = plot_fit_fig(pp, alpha_bc['r'].values ** 3, "alpha_BC-r3", 0, shell ** 3, "m^3")
    alpha_reconr3 = plot_fit_fig(pp, alpha_recon['r'].values ** 3, "alpha_recon-r3", 0, shell ** 3, "m^3")
    ## z-xy fistribution
    plot_zxy(pp, alpha_bc['xy'].values ** 2, alpha_bc['z'], "alpha_bc")
    plot_zxy(pp, alpha_recon['xy'].values ** 2, alpha_recon['z'], "alpha_recon")

    # beta plot
    ## energy distribution
    beta_bcE = plot_fit_fig(pp, beta_bc['E'].values, "beta_BC-E", 0, 10, "MeV")
    beta_reconE = plot_fit_fig(pp, beta_recon['E'].values, "recon-totalcharge", 0, 10, "MeV")
    ## vertex
    ### x
    beta_bcx = plot_fit_fig(pp, beta_bc['x'].values, "beta_BC-x", -shell, shell, "m")
    beta_reconx = plot_fit_fig(pp, beta_recon['x'].values, "beta_recon-x", -shell, shell, "m")
    ### y
    beta_bcy = plot_fit_fig(pp, beta_bc['y'].values, "beta_BC-y", -shell, shell, "m")
    beta_recony = plot_fit_fig(pp, beta_recon['y'].values, "beta_recon-y", -shell, shell, "m")
    ### z
    beta_bcz = plot_fit_fig(pp, beta_bc['z'].values, "beta_BC-z", -shell, shell, "m")
    beta_reconz = plot_fit_fig(pp, beta_recon['z'].values, "beta_recon-z", -shell, shell, "m")
    ### r
    beta_bcr = plot_fit_fig(pp, beta_bc['r'].values, "beta_BC-r", 0, shell, "m")
    beta_reconr = plot_fit_fig(pp, beta_recon['r'].values, "beta_recon-r", 0, shell, "m")
    ### r^3
    beta_bcr3 = plot_fit_fig(pp, beta_bc['r'].values ** 3, "beta_BC-r3", 0, shell ** 3, "m^3")
    beta_reconr3 = plot_fit_fig(pp, beta_recon['r'].values ** 3, "beta_recon-r3", 0, shell ** 3, "m^3")
    ## z-xy fistribution
    plot_zxy(pp, beta_bc['xy'].values ** 2, beta_bc['z'], "beta_bc")
    plot_zxy(pp, beta_recon['xy'].values ** 2, beta_recon['z'], "beta_recon")

    # prompt delayed
    plot_hist2d(pp, alpha_bc['E'].values, beta_bc['E'].values, "bc-alpha energy", "bc-beta energy", 0, 5, 0, 5, "MeV", "MeV", 50)
    plot_hist2d(pp, alpha_recon['E'].values, beta_recon['E'].values, "recon-alpha energy", "recon-beta energy", 0, 5, 0, 5, "MeV", "MeV", 50)

