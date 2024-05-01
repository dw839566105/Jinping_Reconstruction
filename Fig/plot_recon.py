import numpy as np
import tables
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import fit

shell = 0.65
nPE = 65
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
# psr.add_argument('--cut', dest='cut', type=str, default=False, help='recon std cut')
args = psr.parse_args()

def plot_fit(data, title, start, end, unit):
    x, popt, pcov = fit.fitdata(data, start, end, 10)
    fig, ax = plt.subplots()
    ax.hist(data, bins = 100, histtype='step', label=title)
    ax.plot(x, fit.gauss(x, popt[0], popt[1], popt[2], popt[3]), label=f'mu-{popt[0]:.3f} sigma-{popt[1]:.3f}')
    ax.legend()
    ax.set_title(f'{title} Distribution sigma/mu-{(popt[1]/popt[0]):.3f}')
    ax.set_xlabel(f'{title} / {unit}')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)
    return popt[0]

def plot_hist2d(data1, data2, title, start, end, unit):
    fig, ax = plt.subplots()
    h = ax.hist2d(data1, data2, bins = 100, range=[[start, end], [start, end]], cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title(f'{title} Distribution')
    ax.set_xlabel(f'Delayed / {unit}')
    ax.set_ylabel(f'Prompt / {unit}')
    pp.savefig(fig)
    plt.close(fig)

def plot_zxy(data1, data2, title):
    x = np.linspace(0, 0.65 ** 2, 600)
    y = np.sqrt(0.65 ** 2 - x)
    fig, ax = plt.subplots(figsize=(8, 8))
    h = ax.hist2d(data1, data2, bins = 100, range=[[0, 0.65 ** 2], [-0.65, 0.65]], cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.plot(x, y, color='r', linestyle='--')
    ax.plot(x, -y, color='r', linestyle='--')
    ax.set_title(f'{title} scatter z-(x^2+y^2)')
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)

with h5py.File(args.ipt, "r") as f:
    stack = pd.DataFrame(f['stack'][:])
    # step = pd.DataFrame(f['step'][:])
    bc = pd.DataFrame(f['bc'][:])
    fsmp = pd.DataFrame(f['fsmp'][:])
alpha_stack = stack[stack["particle"] == 0]
beta_stack = stack[stack["particle"] == 1]
# alpha_step = step[step["particle"] == 0]
# beta_step = step[step["particle"] == 1]
alpha_bc = bc[bc["particle"] == 0]
beta_bc = bc[bc["particle"] == 1]
alpha_fsmp = fsmp[fsmp["particle"] == 0]
beta_fsmp = fsmp[fsmp["particle"] == 1]

# 排序配对
alpha_stack = alpha_stack.sort_values(by=['EventID'])
beta_stack = beta_stack.sort_values(by=['EventID'])
# alpha_step = alpha_step.sort_values(by=['EventID'])
# beta_step = beta_step.sort_values(by=['EventID'])
alpha_bc = alpha_bc.sort_values(by=['EventID'])
beta_bc = beta_bc.sort_values(by=['EventID'])
alpha_fsmp = alpha_fsmp.sort_values(by=['EventID'])
beta_fsmp = beta_fsmp.sort_values(by=['EventID'])

with PdfPages(args.opt) as pp:
    # alpha plot
    ## energy distribution
    alpha_bcE = plot_fit(alpha_bc['E'].values * nPE, "alpha_BC-PE", 0, 300, "PE")
    alpha_stackE = plot_fit(alpha_stack['E'].values, "alpha_stack-E", 0, 6, "MeV")
    #alpha_stepE = plot_fit(alpha_step['E'].values, "alpha_step-E", 0, 6, "MeV")
    alpha_fsmpE = plot_fit(alpha_fsmp['E'].values, "alpha_fsmp-totalcharge", 0, 300, "PE")
    ## vertex
    ### x
    alpha_bcx = plot_fit(alpha_bc['x'].values, "alpha_BC-x", -0.65, 0.65, "m")
    alpha_stackx = plot_fit(alpha_stack['x'].values, "alpha_stack-x", -0.65, 0.65, "m")
    #alpha_stepx = plot_fit(alpha_step['x'].values, "alpha_step-x", -0.65, 0.65, "m")
    ### y
    alpha_bcy = plot_fit(alpha_bc['y'].values, "alpha_BC-y", -0.65, 0.65, "m")
    alpha_stacky = plot_fit(alpha_stack['y'].values, "alpha_stack-y", -0.65, 0.65, "m")
    #alpha_stepy = plot_fit(alpha_step['y'].values, "alpha_step-y", -0.65, 0.65, "m")
    ### z
    alpha_bcz = plot_fit(alpha_bc['z'].values, "alpha_BC-z", -0.65, 0.65, "m")
    alpha_stackz = plot_fit(alpha_stack['z'].values, "alpha_stack-z", -0.65, 0.65, "m")
    #alpha_stepz = plot_fit(alpha_step['z'].values, "alpha_step-z", -0.65, 0.65, "m")
    ### r
    alpha_bcr = plot_fit(alpha_bc['r'].values, "alpha_BC-r", 0, 0.65, "m")
    alpha_stackr = plot_fit(alpha_stack['r'].values, "alpha_stack-r", 0, 0.65, "m")
    #alpha_stepr = plot_fit(alpha_step['r'].values, "alpha_step-r", 0, 0.65, "m")
    ### r^3
    alpha_bcr3 = plot_fit(alpha_bc['r'].values ** 3, "alpha_BC-r3", 0, 0.65 ** 3, "m^3")
    alpha_stackr3 = plot_fit(alpha_stack['r'].values ** 3, "alpha_stack-r3", 0, 0.65 ** 3, "m^3")
    #alpha_stepr3 = plot_fit(alpha_step['r'].values ** 3, "alpha_step-r3", 0, 0.65 ** 3, "m^3")
    ## z-xy fistribution
    plot_zxy(alpha_bc['x'].values ** 2 + alpha_bc['y'].values ** 2, alpha_bc['z'], "alpha_bc")
    plot_zxy(alpha_stack['x'].values ** 2 + alpha_stack['y'].values ** 2, alpha_stack['z'], "alpha_stack")
    #plot_zxy(alpha_step['x'].values ** 2 + alpha_step['y'].values ** 2, alpha_step['z'], "alpha_step")

    # beta plot
    ## energy distribution
    beta_bcE = plot_fit(beta_bc['E'].values * nPE, "beta_BC-E", 0, 300, "PE")
    beta_stackE = plot_fit(beta_stack['E'].values, "beta_stack-E", 0, 6, "MeV")
    #beta_stepE = plot_fit(beta_step['E'].values, "beta_step-E", 0, 6, "MeV")
    beta_fsmpE = plot_fit(beta_stack['E'].values, "fsmp-totalcharge", 0, 300, "PE")
    ## vertex
    ### x
    beta_bcx = plot_fit(beta_bc['x'].values, "beta_BC-x", -0.65, 0.65, "m")
    beta_stackx = plot_fit(beta_stack['x'].values, "beta_stack-x", -0.65, 0.65, "m")
    #beta_stepx = plot_fit(beta_step['x'].values, "beta_step-x", -0.65, 0.65, "m")
    ### y
    beta_bcy = plot_fit(beta_bc['y'].values, "beta_BC-y", -0.65, 0.65, "m")
    beta_stacky = plot_fit(beta_stack['y'].values, "beta_stack-y", -0.65, 0.65, "m")
    #beta_stepy = plot_fit(beta_step['y'].values, "beta_step-y", -0.65, 0.65, "m")
    ### z
    beta_bcz = plot_fit(beta_bc['z'].values, "beta_BC-z", -0.65, 0.65, "m")
    beta_stackz = plot_fit(beta_stack['z'].values, "beta_stack-z", -0.65, 0.65, "m")
    #beta_stepz = plot_fit(beta_step['z'].values, "beta_step-z", -0.65, 0.65, "m")
    ### r
    beta_bcr = plot_fit(beta_bc['r'].values, "beta_BC-r", 0, 0.65, "m")
    beta_stackr = plot_fit(beta_stack['r'].values, "beta_stack-r", 0, 0.65, "m")
    #beta_stepr = plot_fit(beta_step['r'].values, "beta_step-r", 0, 0.65, "m")
    ### r^3
    beta_bcr3 = plot_fit(beta_bc['r'].values ** 3, "beta_BC-r3", 0, 0.65 ** 3, "m^3")
    beta_stackr3 = plot_fit(beta_stack['r'].values ** 3, "beta_stack-r3", 0, 0.65 ** 3, "m^3")
    #beta_stepr3 = plot_fit(beta_step['r'].values ** 3, "beta_step-r3", 0, 0.65 ** 3, "m^3")
    ## z-xy fistribution
    plot_zxy(beta_bc['x'].values ** 2 + beta_bc['y'].values ** 2, beta_bc['z'], "beta_bc")
    plot_zxy(beta_stack['x'].values ** 2 + beta_stack['y'].values ** 2, beta_stack['z'], "beta_stack")
    #plot_zxy(beta_step['x'].values ** 2 + beta_step['y'].values ** 2, beta_step['z'], "beta_step")

    # prompt delayed
    plot_hist2d(alpha_bc['E'].values * nPE, beta_bc['E'].values * nPE, "Prompt-Delayed (BC)", 0, 300, "PE")
    plot_hist2d(alpha_stack['E'].values, beta_stack['E'].values, "Prompt-Delayed (stack)", 0, 6, "MeV")
    #plot_hist2d(alpha_step['E'].values, beta_step['E'].values, "Prompt-Delayed (step)", 0, 6, "MeV")
    plot_hist2d(alpha_fsmp['E'].values * nPE, beta_fsmp['E'].values * nPE, "Prompt-Delayed (fsmp-totalcharge)", 0, 300, "PE")
