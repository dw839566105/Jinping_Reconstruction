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
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

data = pq.read_table(args.ipt).to_pandas()
E = data['convergence_E'][0]
v = np.stack(data['convergence_v'][0])
t = data['convergence_t'][0]
step = np.arange(2500, 25001, 100)

with PdfPages(args.opt) as pp:
    # 能量变化
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(step, E)
    ax.set_title('Rc_E Evolution')
    ax.set_ylabel('Rc_E')
    ax.set_xlabel('step')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(step, v[:,0])
    ax.set_title('Rc_x Evolution')
    ax.set_ylabel('Rc_x')
    ax.set_xlabel('step')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(step, v[:,1])
    ax.set_title('Rc_y Evolution')
    ax.set_ylabel('Rc_y')
    ax.set_xlabel('step')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(step, v[:,2])
    ax.set_title('Rc_z Evolution')
    ax.set_ylabel('Rc_z')
    ax.set_xlabel('step')
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(step, t)
    ax.set_title('Rc_t Evolution')
    ax.set_ylabel('Rc_t')
    ax.set_xlabel('step')
    pp.savefig(fig)
    plt.close(fig)