import numpy as np
import tables
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("-n", dest="num", type=int, default=1, help="Entries")
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as recon:
    recon = pd.DataFrame(recon['Recon'][:])
breakpoint()
with PdfPages(args.opt) as pp:
    for eid in recon['EventID'].unique()[:args.num]:
        data = recon[recon['EventID'] == eid]
        r = np.sqrt(data['x'].values ** 2 + data['y'].values ** 2 + data['z'].values ** 2)
        
        # 能谱
        fig, ax = plt.subplots()
        ax.hist(data['E'].values, bins = 100, histtype='step')
        ax.set_title(f'Energy Distribution - Event{eid}')
        ax.set_xlabel('Energy / MeV')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        # vertex 分布
        fig, ax = plt.subplots()
        ax.hist(data['x'].values, bins = 100, histtype='step')
        ax.set_title(f'x Distribution - Event{eid}')
        ax.set_xlabel('x / m')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(data['y'].values, bins = 100, histtype='step')
        ax.set_title(f'y Distribution - Event{eid}')
        ax.set_xlabel('y / m')
        ax.set_ylabel('steps')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(data['z'].values, bins = 100, histtype='step')
        ax.set_title(f'z Distribution - Event{eid}')
        ax.set_xlabel('z / m')
        ax.set_ylabel('steps')
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
