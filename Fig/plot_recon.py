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
    bc = pd.DataFrame(f['bc'][:])
alpha_mcmc = mcmc[mcmc["particle"] == 0]
beta_mcmc = mcmc[mcmc["particle"] == 1]
alpha_bc = bc[bc["particle"] == 0]
beta_bc = bc[bc["particle"] == 1]

x_alpha, popt_alpha, pcov_alpha = fit.fitdata(alpha_mcmc['E'].values, 0.22, 0.28, 10)
x_beta, popt_beta, pcov_beta = fit.fitdata(beta_mcmc['E'].values, 0, 1, 10)

with PdfPages(args.opt) as pp:
    fig, ax = plt.subplots()
    ax.hist(alpha_mcmc['E'], bins = 100, histtype='step', label='recon')
    ax.plot(x_alpha, fit.gauss(x_alpha, popt_alpha[0], popt_alpha[1], popt_alpha[2], popt_alpha[3]), label=f'mu-{popt_alpha[0]:.3f} sigma-{popt_alpha[1]:.3f}')
    ax.legend()
    ax.set_title(f'alpha_Energy Distribution sigma/mu-{(popt_alpha[1]/popt_alpha[0]):.3f}')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(alpha_mcmc['xy'], alpha_mcmc['z'], alpha=0.2, s=5, label='recon')
    ax.axvline(x=0.845, color='r', linestyle='--', label='x^2+y^2=0.65^2+0.65^2')
    ax.axhline(y=-0.65, color='g', linestyle='--', label='z=-0.65')
    ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
    ax.set_title('alpha scatter z-average(x^2+y^2)')
    ax.legend()
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(alpha_mcmc['x'].values ** 2 + alpha_mcmc['y'].values ** 2, alpha_mcmc['z'], alpha=0.2, s=5, label='recon')
    ax.axvline(x=0.845, color='r', linestyle='--', label='x^2+y^2=0.65^2+0.65^2')
    ax.axhline(y=-0.65, color='g', linestyle='--', label='z=-0.65')
    ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
    ax.set_title('alpha scatter z-(average(x)^2+average(y)^2)')
    ax.legend()
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)
    '''
    fig, ax = plt.subplots()
    h = ax.hist2d(alpha_mcmc['xy'], alpha_mcmc['z'], bins = 100, cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('alpha hist2d z-(x^2+y^2)')
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)
    '''
    fig, ax = plt.subplots()
    ax.hist(alpha_bc['E'], bins = 100, histtype='step')
    ax.set_title('alpha_Energy(BC) Distribution')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(alpha_mcmc['r'], bins = 100, histtype='step')
    ax.set_title('alpha_r Distribution')
    ax.set_xlabel('r / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.square(alpha_mcmc['r']), bins = 100, histtype='step')
    ax.set_title('alpha_r^3 Distribution')
    ax.set_xlabel('r^3 / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(alpha_bc['r'], bins = 100, histtype='step')
    ax.set_title('alpha_r(BC) Distribution')
    ax.set_xlabel('r / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.square(alpha_bc['r']), bins = 100, histtype='step')
    ax.set_title('alpha_r^3 (BC) Distribution')
    ax.set_xlabel('r^3 / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    # beta
    fig, ax = plt.subplots()
    ax.hist(beta_mcmc['E'], bins = 100, histtype='step', label='recon')
    ax.plot(x_beta, fit.gauss(x_beta, popt_beta[0], popt_beta[1], popt_beta[2],popt_beta[3]), label=f'mu-{popt_beta[0]:.3f} sigma-{popt_beta[1]:.3f}')
    ax.legend()
    ax.set_title(f'beta_Energy Distribution sigma/mu-{(popt_beta[1]/popt_beta[0]):.3f}')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(beta_mcmc['xy'], beta_mcmc['z'], alpha=0.2, s=5, label='recon')
    ax.axvline(x=0.845, color='r', linestyle='--', label='x^2+y^2=0.65^2+0.65^2')
    ax.axhline(y=-0.65, color='g', linestyle='--', label='z=-0.65')
    ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
    ax.set_title('beta scatter z-average(x^2+y^2)')
    ax.legend()
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(beta_mcmc['x'].values ** 2 + beta_mcmc['y'].values ** 2, beta_mcmc['z'], alpha=0.2, s=5, label='recon')
    ax.axvline(x=0.845, color='r', linestyle='--', label='x^2+y^2=0.65^2+0.65^2')
    ax.axhline(y=-0.65, color='g', linestyle='--', label='z=-0.65')
    ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
    ax.set_title('beta scatter z-(average(x)^2+average(y)^2)')
    ax.legend()
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(beta_mcmc['xy'], beta_mcmc['z'], bins = 100, cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('beta hist2d z-(x^2+y^2)')
    ax.set_xlabel('x^2+y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(beta_bc['E'], bins = 100, histtype='step')
    ax.set_title('beta_Energy(BC) Distribution')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(beta_mcmc['r'], bins = 100, histtype='step')
    ax.set_title('beta_r Distribution')
    ax.set_xlabel('r / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.square(beta_mcmc['r']), bins = 100, histtype='step')
    ax.set_title('beta_r^3 Distribution')
    ax.set_xlabel('r^3 / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(beta_bc['r'], bins = 100, histtype='step')
    ax.set_title('beta_r(BC) Distribution')
    ax.set_xlabel('r / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.square(beta_bc['r']), bins = 100, histtype='step')
    ax.set_title('beta_r^3 (BC) Distribution')
    ax.set_xlabel('r^3 / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)