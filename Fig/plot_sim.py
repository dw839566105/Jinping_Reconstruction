import numpy as np
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
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

    # vertex
    ## x
    truthx = plot_fit_fig(pp, truth['x'].values, "truth-x", "Entries", -shell, shell, "m")
    reconx = plot_fit_fig(pp, recon['x'].values, "recon-x", "Entries", -shell, shell, "m")
    ## y
    truthy = plot_fit_fig(pp, truth['y'].values, "truth-y", "Entries", -shell, shell, "m")
    recony = plot_fit_fig(pp, recon['y'].values, "recon-y", "Entries", -shell, shell, "m")
    ## z
    truthz = plot_fit_fig(pp, truth['z'].values, "truth-z", "Entries", -shell, shell, "m")
    reconz = plot_fit_fig(pp, recon['z'].values, "recon-z", "Entries", -shell, shell, "m")
    ## r
    truthr = plot_fit_fig(pp, truth['r'].values, "truth-r", "Entries", -shell, shell, "m")
    reconr = plot_fit_fig(pp, recon['r'].values, "recon-r", "Entries", -shell, shell, "m")
    ## xy
    plot_hist(pp, truth['xy'].values ** 2, "truth-x^2+y^2", "Entries", "m^2")
    plot_hist(pp, recon['xy'].values ** 2, "recon-x^2+y^2", "Entries", "m^2")
    ## r^3
    plot_hist(pp, truth['r'].values ** 3, "truth-r^3", "Entries", "m^3")
    plot_hist(pp, recon['r'].values ** 3, "recon-r^3", "Entries", "m^3")

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

    ## z-xy fistribution
    plot_zxy(pp, truth['xy'].values ** 2, truth['z'], "truth")
    plot_zxy(pp, recon['xy'].values ** 2, recon['z'], "recon")

    plot_scatter(pp, truth['xy'].values ** 2, truth['z'], "truth-xy", "truth-z", "m^2", "m")
    plot_scatter(pp, recon['xy'].values ** 2, recon['z'], "recon-xy", "recon-z", "m^2", "m")

    # recon hist2d
    plot_hist2d(pp, truth['x'].values, recon['x'].values, "truth x", "recon x", -shell, shell, -shell, shell, "m", "m", 50)
    plot_hist2d(pp, truth['y'].values, recon['y'].values, "truth y", "recon y", -shell, shell, -shell, shell, "m", "m", 50)
    plot_hist2d(pp, truth['z'].values, recon['z'].values, "truth z", "recon z", -shell, shell, -shell, shell, "m", "m", 50)
    plot_hist2d(pp, truth['r'].values, recon['r'].values, "truth r", "recon r", -shell, shell, -shell, shell, "m", "m", 50)
    plot_hist2d(pp, truth['xy'].values ** 2, recon['xy'].values ** 2, "truth x^2+y^2", "recon x^2+y^2", 0, shell**2, 0, shell**2, "m^2", "m^2", 50)
    plot_hist2d(pp, truth['r'].values ** 3, recon['r'].values ** 3, "truth r^3", "recon r^3", 0, shell**3, 0, shell**3, "m^3", "m^3", 50)



