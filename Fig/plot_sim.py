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
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as f:
    mcmc = pd.DataFrame(f['mcmc'][:])
    time = pd.DataFrame(f['time'][:])
    truth = pd.DataFrame(f['truth'][:])

def plot_vertex_fit(data, title):
    breakpoint()
    x, popt, pcov = fit.fitdata(data, -0.65, 0.65, 10)
    fig, ax = plt.subplots()
    ax.hist(data, bins = 100, histtype='step', label=title)
    ax.plot(x, fit.gauss(x, popt[0], popt[1], popt[2], popt[3]), label=f'mu-{popt[0]:.3f} sigma-{popt[1]:.3f}')
    ax.legend()
    ax.set_title(f'{title} Distribution sigma/mu-{(popt[1]/popt[0]):.3f}')
    ax.set_xlabel(f'{title} / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

def plot_E_diff(data1, data2, title):
    merged_df = pd.merge(data1, data2, on=['FileNo', 'EventID'], suffixes=('_1', '_2'))
    fig, ax = plt.subplots()
    ax.hist(merged_df['E_1'] - merged_df['E_2'], bins = 100, histtype='step')
    ax.set_title(f'recon E bias - {title}')
    ax.set_xlabel('E / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

def plot_recon_diff(data1, data2, title):
    merged_df = pd.merge(data1, data2, on=['FileNo', 'EventID'], suffixes=('_1', '_2'))
    plot_vertex_fit(merged_df['x_1'] - merged_df['x_2'], "x_" + title)
    plot_vertex_fit(merged_df['y_1'] - merged_df['y_2'], "y_" + title)
    plot_vertex_fit(merged_df['z_1'] - merged_df['z_2'], "z_" + title)

merged_df = pd.merge(mcmc, time, on=['FileNo', 'EventID'], suffixes=('_mcmc', '_time'))

breakpoint()
E_recon, popt_recon, pcov_recon = fit.fitdata(mcmc['E'].values, 0, 4, 10)
E_time, popt_time, pcov_time = fit.fitdata(time['E'].values, 0, 4, 10)
E_truth, popt_truth, pcov_truth = fit.fitdata(truth['E'].values, 0, 300, 10)

with PdfPages(args.opt) as pp:
    fig, ax = plt.subplots()
    ax.hist(mcmc['E'], bins = 100, histtype='step', label='recon_Energy(withoutT)')
    ax.plot(E_recon, fit.gauss(E_recon, popt_recon[0], popt_recon[1], popt_recon[2], popt_recon[3]), label=f'mu-{popt_recon[0]:.3f} sigma-{popt_recon[1]:.3f}')
    ax.legend()
    ax.set_title(f'recon_Energy(withoutT) Distribution sigma/mu-{(popt_recon[1]/popt_recon[0]):.3f}')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(time['E'], bins = 100, histtype='step', label='recon_Energy(withT)')
    ax.plot(E_time, fit.gauss(E_time, popt_time[0], popt_time[1], popt_time[2], popt_time[3]), label=f'mu-{popt_time[0]:.3f} sigma-{popt_time[1]:.3f}')
    ax.legend()
    ax.set_title(f'recon_Energy(withT) Distribution sigma/mu-{(popt_time[1]/popt_time[0]):.3f}')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(truth['E'], bins = 100, histtype='step', label='recon')
    ax.plot(E_truth, fit.gauss(E_truth, popt_truth[0], popt_truth[1], popt_truth[2], popt_truth[3]), label=f'mu-{popt_truth[0]:.3f} sigma-{popt_truth[1]:.3f}')
    ax.legend()
    ax.set_title(f'truth_Energy Distribution sigma/mu-{(popt_truth[1]/popt_truth[0]):.3f}')
    ax.set_xlabel('Energy / NPE')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    plot_E_diff(mcmc, time, "withoutT - withT")

    plot_vertex_fit(mcmc['x'].values, "recon_x(withoutT)")
    plot_vertex_fit(time['x'].values, "recon_x(withT)")
    plot_vertex_fit(truth['x'].values, "truth_x")
    plot_vertex_fit(mcmc['y'].values, "recon_y(withoutT)")
    plot_vertex_fit(time['y'].values, "recon_y(withT)")
    plot_vertex_fit(truth['y'].values, "truth_y")
    plot_vertex_fit(mcmc['z'].values, "recon_z(withoutT)")
    plot_vertex_fit(time['z'].values, "recon_z(withT)")
    plot_vertex_fit(truth['z'].values, "truth_z")

    plot_recon_diff(mcmc, time, "withoutT - withT")
    plot_recon_diff(mcmc, truth, "withoutT - truth")
    plot_recon_diff(time, truth, "withT - truth")

    fig, ax = plt.subplots()
    h = ax.hist2d(mcmc['x'], truth['x'], bins = 100, cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('x(withoutT) distribution')
    ax.set_xlabel('mcmc-x / m')
    ax.set_ylabel('truth-x / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(mcmc['y'], truth['y'], bins = 100, cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('y(withoutT) distribution')
    ax.set_xlabel('mcmc-y / m')
    ax.set_ylabel('truth-y / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(mcmc['z'], truth['z'], bins = 100, range=[[-0.65, 0.65], [-0.65, 0.65]], cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('z(withoutT) distribution')
    ax.set_xlabel('mcmc-z / m')
    ax.set_ylabel('truth-z / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(time['x'], truth['x'], bins = 100, cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('x(withT) distribution')
    ax.set_xlabel('mcmc-x / m')
    ax.set_ylabel('truth-x / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(time['y'], truth['y'], bins = 100, cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('y(withT) distribution')
    ax.set_xlabel('mcmc-y / m')
    ax.set_ylabel('truth-y / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(time['z'], truth['z'], bins = 100, range=[[-0.65, 0.65], [-0.65, 0.65]], cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('z(withT) distribution')
    ax.set_xlabel('mcmc-z / m')
    ax.set_ylabel('truth-z / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(truth['xy'], truth['z'], alpha=0.2, s=5, label='truth')
    ax.axvline(x=0.4225, color='r', linestyle='--', label='x^2+y^2=0.65^2')
    ax.axhline(y=-0.65, color='g', linestyle='--', label='z=-0.65')
    ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
    ax.set_title('truth scatter z-(x^2+y^2)')
    ax.legend()
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mcmc['xy'], mcmc['z'], alpha=0.2, s=5, label='recon')
    ax.axvline(x=0.4225, color='r', linestyle='--', label='x^2+y^2=0.65^2')
    ax.axhline(y=-0.65, color='g', linestyle='--', label='z=-0.65')
    ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
    ax.set_title('recon scatter z-average(x^2+y^2)')
    ax.legend()
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mcmc['x'].values ** 2 + mcmc['y'].values ** 2, mcmc['z'], alpha=0.2, s=5, label='recon(withoutT)')
    ax.axvline(x=0.4225, color='r', linestyle='--', label='x^2+y^2=0.65^2')
    ax.axhline(y=-0.65, color='g', linestyle='--', label='z=-0.65')
    ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
    ax.set_title('recon scatter z-(average(x)^2+average(y)^2)')
    ax.legend()
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(time['x'].values ** 2 + time['y'].values ** 2, time['z'], alpha=0.2, s=5, label='recon(withT)')
    ax.axvline(x=0.4225, color='r', linestyle='--', label='x^2+y^2=0.65^2')
    ax.axhline(y=-0.65, color='g', linestyle='--', label='z=-0.65')
    ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
    ax.set_title('recon scatter z-(average(x)^2+average(y)^2)')
    ax.legend()
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)
    '''
    fig, ax = plt.subplots()
    h = ax.hist2d(mcmc['xy'], mcmc['z'], bins = 100, cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('recon hist2d z-(x^2+y^2)')
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)
    '''
    fig, ax = plt.subplots()
    ax.hist(mcmc['r'], bins = 100, histtype='step', label='recon(withoutT)')
    ax.hist(time['r'], bins = 100, histtype='step', label='recon(withT)')
    ax.hist(truth['r'], bins = 100, histtype='step', label='truth')
    ax.legend()
    ax.set_title('r Distribution')
    ax.set_xlabel('r / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)


    fig, ax = plt.subplots()
    ax.hist(np.array(mcmc['r']) ** 3, bins = 100, histtype='step', label='recon(withoutT)')
    ax.hist(np.array(time['r']) ** 3, bins = 100, histtype='step', label='recon(withT)')
    ax.hist(np.array(truth['r']) ** 3, bins = 100, histtype='step', label='truth')
    ax.legend()
    ax.set_title('recon_r^3 Distribution')
    ax.set_xlabel('r^3 / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

