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
psr.add_argument("ipt", type=str, help="input file")
args = psr.parse_args()

with h5py.File(args.ipt, "r") as recon:
    reconin = pd.DataFrame(recon['ReconIn'][:])
    reconout = pd.DataFrame(recon['ReconOut'][:])
    reconwa = pd.DataFrame(recon['ReconWA'][:]).loc[0]
    reconmcmc = pd.DataFrame(recon['ReconMCMC'][:])
r_reconin = np.sqrt(reconin['x'].values ** 2 + reconin['y'].values ** 2 + reconin['z'].values ** 2)
r_reconout = np.sqrt(reconout['x'].values ** 2 + reconout['y'].values ** 2 + reconout['z'].values ** 2)
r_reconwa = np.sqrt(reconwa['x'] ** 2 + reconwa['y'] ** 2 + reconwa['z'] ** 2) * 0.65
r_reconmcmc = np.sqrt(reconmcmc['x'].values ** 2 + reconmcmc['y'].values ** 2 + reconmcmc['z'].values ** 2)

with PdfPages(args.opt) as pp:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1,1,1)
    fig21 = plt.figure()
    ax21 = fig21.add_subplot(1,1,1)
    fig22 = plt.figure()
    ax22 = fig22.add_subplot(1,1,1)
    fig23 = plt.figure()
    ax23 = fig23.add_subplot(1,1,1)
    fig24 = plt.figure()
    ax24 = fig24.add_subplot(1,1,1)

    # energy_bin = np.linspace(0,0.2,20)

    ax1.hist(reconin['E'].values, bins = 50)
    ax2.hist(reconout['E'].values, bins = 50)
    ax3.hist(reconwa['E'], bins = 50)
    ax4.hist(reconmcmc['E'], bins = 50)
    ax21.hist(r_reconin, bins = 50)
    ax22.hist(r_reconout, bins = 50)
    ax23.hist(r_reconwa, bins = 50)
    ax24.hist(r_reconmcmc, bins = 50)

    ax1.set_title('Energy Distribution (reconin)')
    ax1.set_xlabel('Energy / MeV')
    ax1.set_ylabel('steps')
    ax2.set_title('Energy Distribution (reconout)')
    ax2.set_xlabel('Energy / MeV')
    ax2.set_ylabel('steps')
    ax3.set_title('Energy Distribution (reconwa)')
    ax3.set_xlabel('Energy / MeV')
    ax3.set_ylabel('steps')
    ax4.set_title('Energy Distribution (reconmcmc)')
    ax4.set_xlabel('Energy / MeV')
    ax4.set_ylabel('steps')

    ax21.set_title('r Distribution (reconin)')
    ax21.set_xlabel('r / m')
    ax21.set_ylabel('steps')
    ax22.set_title('r Distribution (reconout)')
    ax22.set_xlabel('r / m')
    ax22.set_ylabel('steps')    
    ax23.set_title('r Distribution (reconwa)')
    ax23.set_xlabel('r / m')
    ax23.set_ylabel('steps')
    ax24.set_title('r Distribution (reconmcmc)')
    ax24.set_xlabel('r / m')
    ax24.set_ylabel('steps')

    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.savefig(fig3)
    pp.savefig(fig4)
    pp.savefig(fig21)
    pp.savefig(fig22)
    pp.savefig(fig23)
    pp.savefig(fig24)

