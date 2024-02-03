'''
pick BiPo events
'''
import pandas as pd
import numpy as np
import uproot
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import fit

psr = argparse.ArgumentParser()
psr.add_argument("-r", dest="recon", help="recon figs")
psr.add_argument("-c", dest="recut", help="recon cut figs")
psr.add_argument("-o", dest='opt', help="BiPo event list output")
psr.add_argument('-i', dest='ipt', help="input")
args = psr.parse_args()

PEs = 65
prompt_s = 0.5
prompt_e = 3.5
delay_s = 0.4
delay_e = 1.2
tcut = 0.0004
dcut = 400
r3cutmin = 0.1
r3cutmax = 0.19
timemax = 1500000
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)

f = uproot.open(args.ipt)
Fold = f['Event']['Fold'].array()
# 时间窗内有两个事例
cut1 = (Fold == 2)
E = f['Event']['E'].array()[cut1]
X = f['Event']['X'].array()[cut1]
Y = f['Event']['Y'].array()[cut1]
Z = f['Event']['Z'].array()[cut1]
TrigSec = f['Event']['TrigSec'].array()[cut1]
TrigNano = f['Event']['TrigNano'].array()[cut1]
TrigNum = f['Event']['TrigNum'].array()[cut1]
FileNum = f['Event']['FileNum'].array()[cut1]
Time = f['Event']['T2PrevSubEvt'].array()[cut1]
D2First = f['Event']['D2First'].array()[cut1]
f.close()

# 能量cut
cut2 = (E[:,0] > prompt_s * PEs) * (E[:,0] < prompt_e * PEs) * (E[:,1] > delay_s * PEs) * (E[:,1] < delay_e * PEs)

# time cut
cut3 = (Time[:,1] < tcut)

# 距离 cut
cut4 = (D2First[:,1] < dcut)

cut = cut2 * cut3

r_alpha = np.sqrt(np.square(X[:,1][cut]) + np.square(Y[:,1][cut]) + np.square(Z[:,1][cut])) / 1000
r_beta = np.sqrt(np.square(X[:,0][cut]) + np.square(Y[:,0][cut]) + np.square(Z[:,0][cut])) / 1000

r3cut_alpha = (r_alpha ** 3 > r3cutmin) * (r_alpha ** 3 < r3cutmax)
r3cut_beta = (r_beta ** 3 > r3cutmin) * (r_beta ** 3 < r3cutmax)

# 生成事例列表
events = len(r_alpha)
alpha = pd.DataFrame({
    'Type': np.repeat('alpha',events),
    'RunNo': np.repeat('0257',events),
    'TrigNo': TrigNum[:,1][cut],
    'FileNo': FileNum[:,1][cut]
})
beta = pd.DataFrame({
    'Type': np.repeat('beta',events),
    'RunNo': np.repeat('0257',events),
    'TrigNo': TrigNum[:,0][cut],
    'FileNo': FileNum[:,0][cut]
})
Bi214_0257 = pd.concat([alpha, beta])
Bi214_0257.to_csv(args.opt, sep=' ', index=False, header=False)

x_alpha, popt_alpha, pcov_alpha = fit.fitdata(np.array(E[:,1][cut] / PEs), 0.7, 1.0, 10)
x_beta, popt_beta, pcov_beta = fit.fitdata(np.array(E[:,0][cut] / PEs), 0, 3.5, 10)

# 画图
with PdfPages(args.recon) as pp:
    fig, ax = plt.subplots()
    ax.hist(E[:,1][cut] / PEs, bins = 100, histtype='step', label='BC')
    ax.plot(x_alpha, fit.gauss(x_alpha, popt_alpha[0], popt_alpha[1], popt_alpha[2], popt_alpha[3]), label=f'mu-{popt_alpha[0]:.3f} sigma-{popt_alpha[1]:.3f}')
    ax.legend()
    ax.set_title(f'alpha_Energy(BC) Distribution sigma/mu-{(popt_alpha[1]/popt_alpha[0]):.3f}')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(E[:,0][cut] / PEs, bins = 100, histtype='step', label='BC')
    ax.plot(x_beta, fit.gauss(x_beta, popt_beta[0], popt_beta[1], popt_beta[2],popt_beta[3]), label=f'mu-{popt_beta[0]:.3f} sigma-{popt_beta[1]:.3f}')
    ax.legend()
    ax.set_title(f'beta_Energy(BC) Distribution sigma/mu-{(popt_beta[1]/popt_beta[0]):.3f}')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(np.array(E[:,1][cut]) / PEs, np.array(E[:,0][cut]) / PEs, bins = 20, cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Energy Distribution')
    ax.set_xlabel('alpha / MeV')
    ax.set_ylabel('beta / MeV')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(r_alpha, bins = 100, histtype='step')
    ax.set_title('alpha_r Distribution')
    ax.set_xlabel('r / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.square(r_alpha), bins = 100, histtype='step')
    ax.set_title('alpha_r^2 Distribution')
    ax.set_xlabel('r^2 / m^2')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(r_alpha ** 3, bins = 100, histtype='step')
    ax.set_title('alpha_r^3 Distribution')
    ax.set_xlabel('r^3 / m^3')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(r_beta, bins = 100, histtype='step')
    ax.set_title('beta_r Distribution')
    ax.set_xlabel('r / m')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.square(r_beta), bins = 100, histtype='step')
    ax.set_title('beta_r^2 Distribution')
    ax.set_xlabel('r^2 / m^2')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(r_beta ** 3, bins = 100, histtype='step')
    ax.set_title('beta_r^3 Distribution')
    ax.set_xlabel('r^3 / m^3')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

# 画图
x_alpha, popt_alpha, pcov_alpha = fit.fitdata(np.array(E[:,1][cut][r3cut_alpha] / PEs), 0.7, 1.0, 10)
x_beta, popt_beta, pcov_beta = fit.fitdata(np.array(E[:,0][cut][r3cut_beta] / PEs), 0, 3.5, 10)

with PdfPages(args.recut) as pp:
    fig, ax = plt.subplots()
    ax.hist(E[:,1][cut][r3cut_alpha] / PEs, bins = 100, histtype='step', label='BC')
    ax.plot(x_alpha, fit.gauss(x_alpha, popt_alpha[0], popt_alpha[1], popt_alpha[2], popt_alpha[3]), label=f'mu-{popt_alpha[0]:.3f} sigma-{popt_alpha[1]:.3f}')
    ax.legend()
    ax.set_title(f'alpha_Energy(BC) Distribution sigma/mu-{(popt_alpha[1]/popt_alpha[0]):.3f}')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(E[:,0][cut][r3cut_beta] / PEs, bins = 100, histtype='step', label='BC')
    ax.plot(x_beta, fit.gauss(x_beta, popt_beta[0], popt_beta[1], popt_beta[2],popt_beta[3]), label=f'mu-{popt_beta[0]:.3f} sigma-{popt_beta[1]:.3f}')
    ax.legend()
    ax.set_title(f'beta_Energy(BC) Distribution sigma/mu-{(popt_beta[1]/popt_beta[0]):.3f}')
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Entries')
    pp.savefig(fig)
    plt.close(fig)
