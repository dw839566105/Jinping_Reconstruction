import numpy as np
import tables
import pandas as pd
import h5py
import fit
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches

jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("-n", dest="num", type=int, default=1, help="Entries")
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as recon:
    recon = pd.DataFrame(recon['Recon'][:])
accept = recon['accept'].sum() / len(recon)
print(f"accept-{accept}")

with PdfPages(args.opt) as pp:
    for eid in recon['EventID'].unique()[:args.num]:
        data = recon[recon['EventID'] == eid]
        r = np.sqrt(data['x'].values ** 2 + data['y'].values ** 2 + data['z'].values ** 2)
        
        # fit
        x_fit, popt_x, pcov_x = fit.fitdata(data['x'].values, -0.65, 0.65, 10)
        y_fit, popt_y, pcov_y = fit.fitdata(data['y'].values, -0.65, 0.65, 10)
        z_fit, popt_z, pcov_z = fit.fitdata(data['z'].values, -0.65, 0.65, 10)
        #E_fit, popt_E, pcov_E = fit.fitdata(data['E'].values, 0, 10000, 10)
        E_fit, popt_E, pcov_E = fit.fitdata(data['E'].values / 2500, 0, 5, 10)

        # 能谱
        fig, ax = plt.subplots()
        ax.hist(data['E'].values / 2500, bins = 100, histtype='step', label='recon')
        # ax.hist(data['E'].values, bins = 100, histtype='step', label='recon')
        ax.plot(E_fit, fit.gauss(E_fit, popt_E[0], popt_E[1], popt_E[2], popt_E[3]), label=f'mu-{popt_E[0]:.3f} sigma-{popt_E[1]:.3f}')
        ax.legend()
        ax.set_title(f'Energy Distribution - Event{eid}')
        ax.set_xlabel('Energy / MeV')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        # vertex 分布
        fig, ax = plt.subplots()
        ax.hist(data['x'].values, bins = 100, histtype='step', label='recon')
        ax.plot(x_fit, fit.gauss(x_fit, popt_x[0], popt_x[1], popt_x[2], popt_x[3]), label=f'mu-{popt_x[0]:.3f} sigma-{popt_x[1]:.3f}')
        ax.legend()
        ax.set_title(f'x Distribution - Event{eid}')
        ax.set_xlabel('x / m')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(data['y'].values, bins = 100, histtype='step', label='recon')
        ax.plot(y_fit, fit.gauss(y_fit, popt_y[0], popt_y[1], popt_y[2], popt_y[3]), label=f'mu-{popt_y[0]:.3f} sigma-{popt_y[1]:.3f}')
        ax.legend()
        ax.set_title(f'y Distribution - Event{eid}')
        ax.set_xlabel('y / m')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(data['z'].values, bins = 100, histtype='step', label='recon')
        ax.plot(z_fit, fit.gauss(z_fit, popt_z[0], popt_z[1], popt_z[2], popt_z[3]), label=f'mu-{popt_z[0]:.3f} sigma-{popt_z[1]:.3f}')
        ax.legend()
        ax.set_title(f'z Distribution - Event{eid}')
        ax.set_xlabel('z / m')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(data['x'].values ** 2 + data['y'].values ** 2, bins = 100, histtype='step')
        ax.set_title(f'x^2+y^2 Distribution - Event{eid}')
        ax.set_xlabel('x^2+y^2 / m^2')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(data['Likelihood'].values, bins = 100, histtype='step')
        ax.set_title(f'Likelihood Distribution - Event{eid}')
        ax.set_xlabel('Likelihood')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        # 位置与 likelihood，二维图
        fig, ax = plt.subplots(figsize=(25, 25))
        h = ax.hist2d(data['z'].values ** 2, data['Likelihood'].values, bins = 50, cmap='Blues')
        fig.colorbar(h[3], ax=ax)
        ax.set_title(f'Likelihood-z Evolution - Event{eid}')
        ax.set_ylabel('Likelihood')
        ax.set_xlabel('z / m')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(25, 25))
        h = ax.hist2d(data['x'].values ** 2 + data['y'].values ** 2, data['Likelihood'].values, bins = 50, cmap='Blues')
        fig.colorbar(h[3], ax=ax)
        ax.set_title(f'Likelihood-x^2+y^2 Evolution - Event{eid}')
        ax.set_ylabel('Likelihood')
        ax.set_xlabel('x^2+y^2 / m^2')
        pp.savefig(fig)
        plt.close(fig)

        # 插值二维图
        fig, ax = plt.subplots(figsize=(25, 25))
        x = data['x'].values ** 2 + data['y'].values ** 2
        y = data['z'].values
        z = data['Likelihood'].values
        xi = np.linspace(0, 0.4225, 100)
        yi = np.linspace(-0.65, 0.65, 100)
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
        ax.contour(xi, yi, zi, levels=14, linewidths=0.1, colors='k')
        cntr1 = ax.contourf(xi, yi, zi, levels=10, cmap="RdBu_r")
        fig.colorbar(cntr1, ax=ax)
        ax.plot(x, y, 'ko', ms=3)
        ax.set(xlim=(0, 0.4225), ylim=(-0.65, 0.65))
        ax.set_title(f'grid and contour {len(x)} points')
        ax.set_xlabel('x^2+y^2 / m^2')
        ax.set_ylabel('z / m')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(25, 25))
        x = data['x'].values ** 2 + data['y'].values ** 2
        y = data['z'].values
        z = data['Likelihood'].values
        xi = np.linspace(0, 0.4225, 100)
        yi = np.linspace(-0.65, 0.65, 100)
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
        ax.contour(xi, yi, zi, levels=14, linewidths=0.1, colors='k')
        cntr1 = ax.contourf(xi, yi, zi, levels=10, cmap="RdBu_r")
        fig.colorbar(cntr1, ax=ax)
        ax.set(xlim=(0, 0.4225), ylim=(-0.65, 0.65))
        ax.set_title(f'grid and contour {len(x)} points')
        ax.set_xlabel('x^2+y^2 / m^2')
        ax.set_ylabel('z / m')
        pp.savefig(fig)
        plt.close(fig)

        
        # 位置分布
        fig, ax = plt.subplots()
        h = ax.scatter(data['x'].values ** 2 + data['y'].values ** 2, data['z'].values, alpha=0.2, s=5, label='recon')
        fig.colorbar(h, ax=ax)
        ax.axvline(x=0.4225, color='r', linestyle='--', label='x^2+y^2=0.65^2')
        impor
        ax.axhline(y=0.65, color='c', linestyle='--', label='z=0.65')
        ax.set_title(f'vertex distribution Event{eid} zstd-{(np.std(y)):.3f} xystd-{(np.std(x)):.3f}')
        ax.set_xlabel('x^2+y^2 / m^2')
        ax.set_ylabel('z / m')
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)
        
        # 能量变化
        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(data['E'].values)
        ax.set_title(f'Energy Evolution - Event{eid}')
        ax.set_ylabel('Energy / MeV')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)

        # 顶点分布
        fig, ax = plt.subplots()
        ax.hist(r, bins = 100, histtype='step')
        ax.set_title(f'r Distribution - Event{eid}')
        ax.set_xlabel('r / m')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        # 体密度分布
        fig, ax = plt.subplots()
        ax.hist(r ** 2, bins = 100, histtype='step')
        ax.set_title(f'r^2 Distribution - Event{eid}')
        ax.set_xlabel('r^2 / m^2')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        # 位置变化
        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(r)
        ax.set_title(f'r Evolution - Event{eid}')
        ax.set_ylabel('r / m')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)

        # x 变化
        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(data['x'].values)
        ax.set_title(f'x Evolution - Event{eid}')
        ax.set_ylabel('x / m')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)

        # y 变化
        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(data['y'].values)
        ax.set_title(f'y Evolution - Event{eid}')
        ax.set_ylabel('y / m')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)

        # z 变化
        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(data['z'].values)
        ax.set_title(f'z Evolution - Event{eid}')
        ax.set_ylabel('z / m')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)

        # 能量-位置散点图
        fig, ax = plt.subplots()
        ax.scatter(r, data['E'].values, alpha=0.1)
        ax.set_title(f'E - r Distribution - Event{eid}')
        ax.set_xlabel('r / m')
        ax.set_ylabel('Energy / MeV')  
        pp.savefig(fig)
        plt.close(fig)

        # 似然函数值变化
        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(data['Likelihood'].values)
        ax.set_title(f'Likelihood Evolution - Event{eid}')
        ax.set_ylabel('Likelihood')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)
