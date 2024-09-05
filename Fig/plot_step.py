import numpy as np
import tables
import pandas as pd
import uproot
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
psr.add_argument("-t", dest="truth", type=str, default=None, help="truth file")
psr.add_argument("--switch", dest="switch", type=str, help="switch")
psr.add_argument("--mode", dest="mode", type=str, help="data mode")
psr.add_argument("-n", dest="num", type=int, default=1, help="Entries")
psr.add_argument("-s", dest="step", type=int, help="MC step")
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as f:
    recon = f['Recon'][:args.num, :]

if args.mode == "sim":
    with uproot.open(args.truth) as ipt:
        info = ipt['SimTriggerInfo']
        PMTID = info['PEList.PMTId'].array()
        x = np.array(info['truthList.x'].array() / 1000).flatten()
        y = np.array(info['truthList.y'].array() / 1000).flatten()
        z = np.array(info['truthList.z'].array() / 1000).flatten()
        r = np.sqrt(x**2 + y**2 + z**2)
        xy = np.sqrt(x**2 + y**2)
        TriggerNo = np.array(info['TriggerNo'].array()).flatten()
        truth = pd.DataFrame({"EventID":TriggerNo, "x":x, "y":y, "z":z, "E":2*np.ones(len(x)), "r":r, "xy":xy})

with PdfPages(args.opt) as pp:
    for data in recon:
        eid = data[0]['EventID']
        if args.mode == "sim":
            truth_data = truth[truth['EventID'] == eid]
        acceptz = data['acceptz'].sum() / data.shape[0]
        acceptr = data['acceptr'].sum() / data.shape[0]
        acceptE = data['acceptE'].sum() / data.shape[0]
        acceptt = data['acceptt'].sum() / data.shape[0]
        print(f"eid-{eid}")
        print(f"acceptz-{acceptz}")
        print(f"acceptr-{acceptr}")
        print(f"acceptE-{acceptE}")
        print(f"acceptt-{acceptt}")
        data_r = np.sqrt(data['x'] ** 2 + data['y'] ** 2 +  data['z'] ** 2) * shell
        data_xy = np.sqrt(data['x'] ** 2 + data['y'] ** 2) * shell
        
        if args.switch == "ON":
            # loglikelihood distribution
            plot_hist(pp, data['Likelihood'], "LogLikelihood", "steps", "value")

            # 插值二维图
            fig, ax = plt.subplots(figsize=(25, 25))
            x = data_xy
            y = data['z'] * shell
            z = data['Likelihood']
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
            y = data['z'] * shell
            z = data['Likelihood']
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
        plot_zxy(pp, data_xy ** 2, data['z'], "recon")

        # Evolution
        fig, axs = plt.subplots(7, 1, figsize=(25, 35))
        axs[0].plot(data['E'])
        if args.mode == "sim":
            axs[0].axhline(y=truth_data['E'], color='g', linestyle='--')
        axs[0].set_title(f'Energy Evolution - Event{eid}')
        axs[0].set_ylabel('Energy / MeV')
        axs[0].set_xlabel('step')

        axs[1].plot(data['x'] * shell)
        if args.mode == "sim":
            axs[1].axhline(y=truth_data['x'], color='g', linestyle='--')
        axs[1].set_title(f'x Evolution - Event{eid}')
        axs[1].set_ylabel('x / m')
        axs[1].set_xlabel('step')

        axs[2].plot(data['y'] * shell)
        if args.mode == "sim":
            axs[2].axhline(y=truth_data['y'], color='g', linestyle='--')
        axs[2].set_title(f'y Evolution - Event{eid}')
        axs[2].set_ylabel('y / m')
        axs[2].set_xlabel('step')

        axs[3].plot(data_r)
        if args.mode == "sim":
            axs[3].axhline(y=np.sqrt(truth_data['x'] ** 2 + truth_data['y'] ** 2 +  + truth_data['z'] ** 2), color='g', linestyle='--')
        axs[3].set_title(f'r Evolution - Event{eid}')
        axs[3].set_ylabel('r / m')
        axs[3].set_xlabel('step')

        axs[4].plot(data['z'] * shell)
        if args.mode == "sim":
            axs[4].axhline(y=truth_data['z'], color='g', linestyle='--')
        axs[4].set_title(f'z Evolution - Event{eid}')
        axs[4].set_ylabel('z / m')
        axs[4].set_xlabel('step')

        axs[5].plot(data['t'])
        axs[5].set_title(f't Evolution - Event{eid}')
        axs[5].set_ylabel('t / ns')
        axs[5].set_xlabel('step')

        axs[6].plot(data['Likelihood'])
        axs[6].set_title(f'LogLikelihood Evolution - Event{eid}')
        axs[6].set_ylabel('LogLikelihood')
        axs[6].set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)
