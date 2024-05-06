import numpy as np
import tables
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import fit

shell = 0.65
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
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
    ax.set_xlabel(f'recon / {unit}')
    ax.set_ylabel(f'truth / {unit}')
    pp.savefig(fig)
    plt.close(fig)

def plot_zxy(data1, data2, title):
    x = np.linspace(0, 0.65 ** 2, 600)
    y = np.sqrt(0.65 ** 2 - x)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(data1, data2, alpha=0.2, s=5)
    ax.plot(x, y, color='r', linestyle='--')
    ax.plot(x, -y, color='r', linestyle='--')
    ax.set_title(f'{title} scatter z-(x^2+y^2)')
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)

with h5py.File(args.ipt, "r") as f:
    truth = pd.DataFrame(f['truth'][:])
    bc = pd.DataFrame(f['bc'][:])
    stack = pd.DataFrame(f['stack'][:])
    stackt = pd.DataFrame(f['stackt'][:])
    step = pd.DataFrame(f['step'][:])
    stept = pd.DataFrame(f['stept'][:])

# 按照 FileNo, EventID 排序
truth = truth.sort_values(by=['FileNo', 'EventID'])
bc = bc.sort_values(by=['FileNo', 'EventID'])
stack = stack.sort_values(by=['FileNo', 'EventID'])
stackt = stackt.sort_values(by=['FileNo', 'EventID'])
step = step.sort_values(by=['FileNo', 'EventID'])
stept = stept.sort_values(by=['FileNo', 'EventID'])
 
with PdfPages(args.opt) as pp:
    # 能量
    truthE = plot_fit(truth['E'].values, "truth-E", 50, 250, "NPE")
    bcE = plot_fit(bc['E'].values, "BC-E", 0, 4, "MeV")
    stackE = plot_fit(stack['E'].values, "stack-E", 0, 4, "MeV")
    stacktE = plot_fit(stackt['E'].values, "stackt-E", 0, 4, "MeV")
    stepE = plot_fit(step['E'].values, "step-E", 0, 4, "MeV")
    steptE = plot_fit(stept['E'].values, "stept-E", 0, 4, "MeV")

    # 能量 bias
    E_bias_stack = plot_fit(stackt['E'].values - stack['E'].values, "bias-E (stackt-stack)", -2, 2, "MeV")
    E_bias_step = plot_fit(stept['E'].values - step['E'].values, "bias-E (stept-step)", -2, 2, "MeV")
    E_bias_ss = plot_fit(stept['E'].values - stackt['E'].values, "bias-E (stept-stackt)", -2, 2, "MeV")
    E_bias_Tstack = plot_fit(stackt['E'].values / stacktE - truth['E'].values / truthE, "bias-scaledE (stackt-truth)", -2, 2, "Energy")
    E_bias_Tstep = plot_fit(stept['E'].values / steptE - truth['E'].values / truthE, "bias-scaledE (stept-truth)", -2, 2, "Energy")
    E_bias_Tbc = plot_fit(bc['E'].values / bcE - truth['E'].values / truthE, "bias-scaledE (BC-truth)", -2, 2, "Energy")

    # vertex
    ## x
    truthx = plot_fit(truth['x'].values, "truth-x", -0.65, 0.65, "m")
    bcx = plot_fit(bc['x'].values, "BC-x", -0.65, 0.65, "m")
    stackx = plot_fit(stack['x'].values, "stack-x", -0.65, 0.65, "m")
    stacktx = plot_fit(stackt['x'].values, "stackt-x", -0.65, 0.65, "m")
    stepx = plot_fit(step['x'].values, "step-x", -0.65, 0.65, "m")
    steptx = plot_fit(stept['x'].values, "stept-x", -0.65, 0.65, "m")
    ## y
    truthy = plot_fit(truth['y'].values, "truth-y", -0.65, 0.65, "m")
    bcy = plot_fit(bc['y'].values, "BC-y", -0.65, 0.65, "m")
    stacky = plot_fit(stack['y'].values, "stack-y", -0.65, 0.65, "m")
    stackty = plot_fit(stackt['y'].values, "stackt-y", -0.65, 0.65, "m")
    stepy = plot_fit(step['y'].values, "step-y", -0.65, 0.65, "m")
    stepty = plot_fit(stept['y'].values, "stept-y", -0.65, 0.65, "m")
    ## z
    truthz = plot_fit(truth['z'].values, "truth-z", -0.65, 0.65, "m")
    bcz = plot_fit(bc['z'].values, "BC-z", -0.65, 0.65, "m")
    stackz = plot_fit(stack['z'].values, "stack-z", -0.65, 0.65, "m")
    stacktz = plot_fit(stackt['z'].values, "stackt-z", -0.65, 0.65, "m")
    stepz = plot_fit(step['z'].values, "step-z", -0.65, 0.65, "m")
    steptz = plot_fit(stept['z'].values, "stept-z", -0.65, 0.65, "m")
    ## r
    truthr = plot_fit(truth['r'].values, "truth-r", 0, 0.65, "m")
    bcr = plot_fit(bc['r'].values, "BC-r", 0, 0.65, "m")
    stackr = plot_fit(stack['r'].values, "stack-r", 0, 0.65, "m")
    stacktr = plot_fit(stackt['r'].values, "stackt-r", 0, 0.65, "m")
    stepr = plot_fit(step['r'].values, "step-r", 0, 0.65, "m")
    steptr = plot_fit(stept['r'].values, "stept-r", 0, 0.65, "m")
    ## r^3
    truthr3 = plot_fit(truth['r'].values ** 3, "truth-r3", 0, 0.65 ** 3, "m^3")
    bcr3 = plot_fit(bc['r'].values ** 3, "BC-r3", 0, 0.65 ** 3, "m^3")
    stackr3 = plot_fit(stack['r'].values ** 3, "stack-r3", 0, 0.65 ** 3, "m^3")
    stacktr3 = plot_fit(stackt['r'].values ** 3, "stackt-r3", 0, 0.65 ** 3, "m^3")
    stepr3 = plot_fit(step['r'].values ** 3, "step-r3", 0, 0.65 ** 3, "m^3")
    steptr3 = plot_fit(stept['r'].values ** 3, "stept-r3", 0, 0.65 ** 3, "m^3")

    # vertex bias
    ## x bias
    x_bias_stack = plot_fit(stackt['x'].values - stack['x'].values, "bias-x (stackt-stack)", -0.65, 0.65, "m")
    x_bias_step = plot_fit(stept['x'].values - step['x'].values, "bias-x (stept-step)", -0.65, 0.65, "m")
    x_bias_ss = plot_fit(stept['x'].values - stackt['x'].values, "bias-x (stept-stackt)", -0.65, 0.65, "m")
    x_bias_Tstack = plot_fit(stackt['x'].values - truth['x'].values, "bias-x (stackt-truth)", -0.65, 0.65, "m")
    x_bias_Tstep = plot_fit(stept['x'].values - truth['x'].values, "bias-x (stept-truth)", -0.65, 0.65, "m")
    x_bias_Tbc = plot_fit(bc['x'].values - truth['x'].values, "bias-x (BC-truth)", -0.65, 0.65, "m") 
    ## y bias
    y_bias_stack = plot_fit(stackt['y'].values - stack['y'].values, "bias-y (stackt-stack)", -0.65, 0.65, "m")
    y_bias_step = plot_fit(stept['y'].values - step['y'].values, "bias-y (stept-step)", -0.65, 0.65, "m")
    y_bias_ss = plot_fit(stept['y'].values - stackt['y'].values, "bias-y (stept-stackt)", -0.65, 0.65, "m")
    y_bias_Tstack = plot_fit(stackt['y'].values - truth['y'].values, "bias-y (stackt-truth)", -0.65, 0.65, "m")
    y_bias_Tstep = plot_fit(stept['y'].values - truth['y'].values, "bias-y (stept-truth)", -0.65, 0.65, "m")
    y_bias_Tbc = plot_fit(bc['y'].values - truth['y'].values, "bias-y (BC-truth)", -0.65, 0.65, "m")
    ## z bias
    z_bias_stack = plot_fit(stackt['z'].values - stack['z'].values, "bias-z (stackt-stack)", -0.65, 0.65, "m")
    z_bias_step = plot_fit(stept['z'].values - step['z'].values, "bias-z (stept-step)", -0.65, 0.65, "m")
    z_bias_ss = plot_fit(stept['z'].values - stackt['z'].values, "bias-z (stept-stackt)", -0.65, 0.65, "m")
    z_bias_Tstack = plot_fit(stackt['z'].values - truth['z'].values, "bias-z (stackt-truth)", -0.65, 0.65, "m")
    z_bias_Tstep = plot_fit(stept['z'].values - truth['z'].values, "bias-z (stept-truth)", -0.65, 0.65, "m")
    z_bias_Tbc = plot_fit(bc['z'].values - truth['z'].values, "bias-z (BC-truth)", -0.65, 0.65, "m")
    ## r bias
    r_bias_stack = plot_fit(stackt['r'].values - stack['r'].values, "bias-r (stackt-stack)", 0, 0.65, "m")
    r_bias_step = plot_fit(stept['r'].values - step['r'].values, "bias-r (stept-step)", 0, 0.65, "m")
    r_bias_ss = plot_fit(stept['r'].values - stackt['r'].values, "bias-r (stept-stackt)", 0, 0.65, "m")
    r_bias_Tstack = plot_fit(stackt['r'].values - truth['r'].values, "bias-r (stackt-truth)", 0, 0.65, "m")
    r_bias_Tstep = plot_fit(stept['r'].values - truth['r'].values, "bias-r (stept-truth)", 0, 0.65, "m")
    r_bias_Tbc = plot_fit(bc['r'].values - truth['r'].values, "bias-r (BC-truth)", 0, 0.65, "m")

    # z-(x^2+y^2)
    plot_zxy(truth['x'].values ** 2 + truth['y'].values ** 2, truth['z'], "truth")
    plot_zxy(bc['x'].values ** 2 + bc['y'].values ** 2, bc['z'], "bc")
    plot_zxy(stack['x'].values ** 2 + stack['y'].values ** 2, stack['z'], "stack")
    plot_zxy(stackt['x'].values ** 2 + stackt['y'].values ** 2, stackt['z'], "stackt")
    plot_zxy(step['x'].values ** 2 + step['y'].values ** 2, step['z'], "step")
    plot_zxy(stept['x'].values ** 2 + stept['y'].values ** 2, stept['z'], "stept")

    # hist2d distribution
    plot_hist2d(stackt['E'].values / stacktE, truth['E'].values / truthE, "scaledE (stackt-truth)", -2, 2, "Energy")
    plot_hist2d(stept['E'].values / steptE, truth['E'].values / truthE, "scaledE (stept-truth)", -2, 2, "Energy")
    plot_hist2d(bc['E'].values / bcE, truth['E'].values / truthE, "scaledE (BC-truth)", -2, 2, "Energy")
    plot_hist2d(stackt['x'].values, truth['x'].values, "x (stackt-truth)", -0.65, 0.65, "m")
    plot_hist2d(stept['x'].values, truth['x'].values, "x (stept-truth)", -0.65, 0.65, "m")
    plot_hist2d(bc['x'].values, truth['x'].values, "x (BC-truth)", -0.65, 0.65, "m")
    plot_hist2d(stackt['y'].values, truth['y'].values, "y (stackt-truth)", -0.65, 0.65, "m")
    plot_hist2d(stept['y'].values, truth['y'].values, "y (stept-truth)", -0.65, 0.65, "m")
    plot_hist2d(bc['y'].values, truth['y'].values, "y (BC-truth)", -0.65, 0.65, "m")
    plot_hist2d(stackt['z'].values, truth['z'].values, "z (stackt-truth)", -0.65, 0.65, "m")
    plot_hist2d(stept['z'].values, truth['z'].values, "z (stept-truth)", -0.65, 0.65, "m")
    plot_hist2d(bc['z'].values, truth['z'].values, "z (BC-truth)", -0.65, 0.65, "m")
    plot_hist2d(stackt['r'].values, truth['r'].values, "r (stackt-truth)", 0, 0.65, "m")
    plot_hist2d(stept['r'].values, truth['r'].values, "r (stept-truth)", 0, 0.65, "m")
    plot_hist2d(bc['r'].values, truth['r'].values, "r (BC-truth)", 0, 0.65, "m")
    plot_hist2d(stackt['r'].values ** 3, truth['r'].values ** 3, "r^3 (stackt-truth)", 0, 0.65 ** 3, "m^3")
    plot_hist2d(stept['r'].values ** 3, truth['r'].values ** 3, "r^3 (stept-truth)", 0, 0.65 ** 3, "m^3")
    plot_hist2d(bc['r'].values ** 3, truth['r'].values ** 3, "r^3 (BC-truth)", 0, 0.65 ** 3, "m^3")
