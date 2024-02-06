import numpy as np
import tables
import pandas as pd
import pyarrow.parquet as pq
import h5py
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("-n", dest="num", type=int, default=1, help="Entries")
psr.add_argument('-m', '--MCstep', dest='MCstep', type=int, default=10, help='mcmc step per PEt')
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

gel = pq.read_table(args.ipt).to_pandas()
step = np.arange(2500, 2500 * args.MCstep + 1, 100)
with PdfPages(args.opt) as pp:
    for eid in gel['EventID'].unique()[:args.num]:
        data = gel[gel['EventID'] == eid]
        E = np.stack(data['convergence_E'][0])
        v = data['convergence_v'][0]
        t = np.stack(data['convergence_t'][0])

        # 能量变化
        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(step, E[:,0])
        if np.max(E[:,0]) > 1.1:
            ax.axhline(y=1.1, color='g', linestyle='--')
        ax.set_title(f'psrf_E Evolution - Event{eid}')
        ax.set_ylabel('psrf_E')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)

        # 位置变化
        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(step, v)
        if np.max(v) > 1.1:
            ax.axhline(y=1.1, color='g', linestyle='--')
        ax.set_title(f'mpsrf_v Evolution - Event{eid}')
        ax.set_ylabel('mpsrf_v')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(25, 5))
        ax.plot(step, t[:,0])
        if np.max(t[:,0]) > 1.1:
            ax.axhline(y=1.1, color='g', linestyle='--')
        ax.set_title(f'psrf_t Evolution - Event{eid}')
        ax.set_ylabel('psrf_t')
        ax.set_xlabel('step')
        pp.savefig(fig)
        plt.close(fig)