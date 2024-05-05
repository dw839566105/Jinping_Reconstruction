import numpy as np
import tables
import pandas as pd
import h5py
import fit
import argparse
import matplotlib.pyplot as plt
from plot_basis import plot_fit, plot_zxy, plot_hist
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
import matplotlib.tri as tri
from DetectorConfig import shell

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("-t", dest="truth", type=str, default=None, help="output file")
psr.add_argument("--switch", dest="switch", type=str, help="switch")
psr.add_argument("--mode", dest="mode", type=str, help="data mode")
psr.add_argument("-n", dest="num", type=int, default=1, help="Entries")
psr.add_argument("-s", dest="step", type=int, help="MC step")
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as f:
    recon = pd.DataFrame(f['Recon'][:])

if args.mode == "sim":
    with h5py.File(args.truth, "r") as f:
        truth = pd.DataFrame(f['vertex'][:])

with PdfPages(args.opt) as pp:
    for eid in recon['EventID'].unique()[:args.num]:
        data = recon[recon['EventID'] == eid]
        if args.mode == "sim":
            truth_data = truth[truth['EventID'] == eid]
        acceptz = data['acceptz'].sum() / data.shape[0]
        acceptr = data['acceptr'].sum() / data.shape[0]
        acceptE = data['acceptE'].sum() / data.shape[0]
        print(f"eid-{eid}")
        print(f"acceptz-{acceptz}")
        print(f"acceptr-{acceptr}")
        print(f"acceptE-{acceptE}")
        data_r = np.sqrt(data['x'].values ** 2 + data['y'].values ** 2 +  data['z'].values ** 2) * shell
        data_xy = np.sqrt(data['x'].values ** 2 + data['y'].values ** 2) * shell
        
        if args.switch == "ON":
            # E distribution
            fig, ax = plt.subplots()
            popt_E = plot_fit(data['E'].values, ax, "E", "steps", 0, 10, "MeV")
            pp.savefig(fig)
            plt.close(fig)

            # vertex distribution
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            popt_x = plot_fit(data['x'].values * shell, axs[0,0], "x", "steps", -shell, shell, "m")
            popt_y = plot_fit(data['y'].values * shell, axs[0,1], "y", "steps", -shell, shell, "m")
            popt_r = plot_fit(data_r, axs[1,0], "r", "steps", 0, shell, "m")
            popt_z = plot_fit(data['z'].values * shell, axs[1,1], "z", "steps", -shell, shell, "m")
            pp.savefig(fig)
            plt.close(fig)

            # r3 distribution
            plot_hist(pp, data_r ** 3, "r^3", "steps", "m^3")

            # t distribution
            plot_hist(pp, data['t'].values, "t", "steps", "ns")

            # loglikelihood distribution
            plot_hist(pp, data['Likelihood'].values, "LogLikelihood", "steps", "value")

            # 插值二维图
            fig, ax = plt.subplots(figsize=(25, 25))
            x = data_xy
            y = data['z'].values * shell
            z = data['Likelihood'].values
            xi = np.linspace(0, shell, 100)
            yi = np.linspace(-shell, shell, 100)
            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            ax.contour(xi, yi, zi, levels=14, linewidths=0.1, colors='k')
            cntr1 = ax.contourf(xi, yi, zi, levels=10, cmap="RdBu_r")
            fig.colorbar(cntr1, ax=ax)
            ax.plot(x, y, 'ko', ms=3)
            ax.set(xlim=(0, shell), ylim=(-shell, shell))
            ax.set_title(f'grid and contour {len(x)} points')
            ax.set_xlabel('sqrt(x^2 + y^2) / m')
            ax.set_ylabel('z / m')
            pp.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(25, 25))
            x = data_xy
            y = data['z'].values * shell
            z = data['Likelihood'].values
            xi = np.linspace(0, shell, 100)
            yi = np.linspace(-shell, shell, 100)
            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            ax.contour(xi, yi, zi, levels=14, linewidths=0.1, colors='k')
            cntr1 = ax.contourf(xi, yi, zi, levels=10, cmap="RdBu_r")
            fig.colorbar(cntr1, ax=ax)
            ax.set(xlim=(0, shell), ylim=(-shell, shell))
            ax.set_title(f'grid and contour {len(x)} points')
            ax.set_xlabel('sqrt(x^2 + y^2) / m')
            ax.set_ylabel('z / m')
            pp.savefig(fig)
            plt.close(fig)
        
        # z-xy 分布
        plot_zxy(pp, data_xy ** 2, data['z'].values)

        # Evolution
        fig, axs = plt.subplots(7, 1, figsize=(25, 35))
        axs[0].plot(data['E'].values)
        if args.mode == "sim":
            axs[0].axhline(y=truth_data['E'].values, color='g', linestyle='--')
        axs[0].set_title(f'Energy Evolution - Event{eid}')
        axs[0].set_ylabel('Energy / MeV')
        axs[0].set_xlabel('step')

        axs[1].plot(data['x'].values * shell)
        if args.mode == "sim":
            axs[1].axhline(y=truth_data['x'].values, color='g', linestyle='--')
        axs[1].set_title(f'x Evolution - Event{eid}')
        axs[1].set_ylabel('x / m')
        axs[1].set_xlabel('step')

        axs[2].plot(data['y'].values * shell)
        if args.mode == "sim":
            axs[2].axhline(y=truth_data['y'].values, color='g', linestyle='--')
        axs[2].set_title(f'y Evolution - Event{eid}')
        axs[2].set_ylabel('y / m')
        axs[2].set_xlabel('step')

        axs[3].plot(data_r)
        if args.mode == "sim":
            axs[3].axhline(y=np.sqrt(truth_data['x'].values ** 2 + truth_data['y'].values ** 2 +  + truth_data['z'].values ** 2), color='g', linestyle='--')
        axs[3].set_title(f'r Evolution - Event{eid}')
        axs[3].set_ylabel('r / m')
        axs[3].set_xlabel('step')

        axs[4].plot(data['z'].values * shell)
        if args.mode == "sim":
            axs[4].axhline(y=truth_data['z'].values, color='g', linestyle='--')
        axs[4].set_title(f'z Evolution - Event{eid}')
        axs[4].set_ylabel('z / m')
        axs[4].set_xlabel('step')

        axs[5].plot(data['t'].values)
        if args.mode == "sim":
            axs[5].axhline(y=truth_data['t'].values, color='g', linestyle='--')
        axs[5].set_title(f't Evolution - Event{eid}')
        axs[5].set_ylabel('t / ns')
        axs[5].set_xlabel('step')

        axs[6].plot(data['Likelihood'].values)
        axs[6].set_title(f'LogLikelihood Evolution - Event{eid}')
        axs[6].set_ylabel('LogLikelihood')
        axs[6].set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)
