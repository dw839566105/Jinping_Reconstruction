import numpy as np
import tables
import pandas as pd
from polynomial import *
from zernike import RZern
from numpy.polynomial import legendre as LG
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
from tqdm import tqdm
import uproot
import awkward as ak

from numba import jit
from polynomial import *

mpl.rcParams['lines.markersize'] = 2
def loadh5(filename):
    h = tables.open_file(filename)
    coef = h.root.coeff[:]
    h.close()
    return coef

def template(vertex, coef, PMT_pos):

    cut = len(coef)
    cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)
    rhof = np.linalg.norm(vertex) + np.zeros(len(PMT_pos))
    r_basis = legval_raw(rhof, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
    t_basis = legval_raw(cos_theta, np.eye(cut).reshape((cut,cut,1))).T
    expect = np.exp((t_basis*r_basis).sum(-1))
    return expect

def tract(h):
    index = h.root.ReconIn[:]['Likelihood'] < h.root.ReconOut[:]['Likelihood'] 
    x = h.root.ReconIn[:]['x']*0.638/0.65
    x[~index] = h.root.ReconOut[:]['x'][~index]*0.638/0.65
    y = h.root.ReconIn[:]['y']*0.638/0.65
    y[~index] = h.root.ReconOut[:]['y'][~index]*0.638/0.65
    z = h.root.ReconIn[:]['z']*0.638/0.65
    z[~index] = h.root.ReconOut[:]['z'][~index]*0.638/0.65
    E = h.root.ReconIn[:]['E']
    E[~index] = h.root.ReconOut[:]['E'][~index]
    xt = h.root.Truth[:]['x']/1000
    yt = h.root.Truth[:]['y']/1000
    zt = h.root.Truth[:]['z']/1000
    Et = h.root.Truth[:]['E']
    return x, y, z, E, xt, yt, zt, Et

with PdfPages('Energy.pdf') as pp:    
    for prefix in ['/mnt/stage/douwei/Upgrade/0.90', '/mnt/stage/douwei/JP_1t_paper']:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        for axis in ['x','z']:
            path = '%s/recon/point/%s/2' % (prefix, axis)
            data = []
            for radius in np.arange(0.01, 0.65, 0.01):
                with tables.open_file('%s/%.2f.h5' % (path, radius)) as h:
                    v = tract(h)
                    h.close()
                data.append(np.vstack((v)).T)
            mean = []
            std = []
            c_mean = []
            c_std = []
            for idx, i in enumerate(data):
                r = np.linalg.norm(i[:,:3], axis=1)
                mean.append((i[:,3]*i[:,-1]).mean())
                c_mean.append((i[r<0.6, 3]*i[r<0.6, -1]).mean())
                std.append((i[:,3]*i[:,-1]).std())
                c_std.append((i[r<0.6, 3]*i[r<0.6, -1]).std())
            mean = np.array(mean)
            std = np.array(std)
            c_mean = np.array(c_mean)
            c_std = np.array(c_std)
            if(axis == 'x'):
                ax1.scatter(np.arange(0.01, 0.65, 0.01)*1000, mean, color='r')
                ax1.fill_between(np.arange(0.01, 0.65, 0.01)*1000, mean - std, mean + std, color='r', alpha=0.1,
                                 label='Fit(ReconV)')
                # ax1.scatter(np.arange(0.01, 0.65, 0.01)[:-5]*1000, c_mean[:-5], color='r')
                # ax1.fill_between(np.arange(0.01, 0.65, 0.01)[:-5]*1000, 
                #                 c_mean[:-5] - c_std[:-5], 
                #                 c_mean[:-5] + c_std[:-5], 
                #                 color='r', alpha=0.5, label='Cut')
            elif axis == 'z':
                ax2.scatter(np.arange(0.01, 0.65, 0.01)*1000, mean, color='r')
                ax2.fill_between(np.arange(0.01, 0.65, 0.01)*1000, mean - std, mean + std, color='r', alpha=0.1,
                                label='Fit(ReconV)')
                # ax2.scatter(np.arange(0.01, 0.65, 0.01)[:-5]*1000, c_mean[:-5], color='r')
                # ax2.fill_between(np.arange(0.01, 0.65, 0.01)[:-5]*1000, 
                #                 c_mean[:-5] - c_std[:-5], 
                #                 c_mean[:-5] + c_std[:-5], 
                #                 color='r', alpha=0.5, label='Cut')
            if prefix == '/mnt/stage/douwei/Upgrade/0.90':
                PMT = np.loadtxt('/home/douwei/UpgradeJP/DetectorStructure/1t_0.90/PMT.txt')
                coef = loadh5('/mnt/stage/douwei/Upgrade/0.90/coeff/Legendre/Gather/PE/2/40/25.h5')
                nPMT = int(121)
            else:
                PMT = np.loadtxt('/home/douwei/ReconJP/PMT.txt')
                coef = loadh5('/mnt/stage/douwei/JP_1t_paper/coeff/Legendre/Gather/PE/2/40/25.h5')
                nPMT = int(31)

            path = '%s/root/point/%s/2' % (prefix, axis)

            mean = []
            std = []
            e_mean = []
            e_std = []

            for radius in tqdm(np.arange(0.01, 0.65, 0.01)):
                with uproot.open('%s/%.2f.root' % (path, radius)) as f:
                    d = f['SimTriggerInfo']
                    pid = d['PEList.PMTId'].array()
                    a = np.array([len(i) for i in pid])
                    mean.append(np.mean(a))
                    std.append(np.std(a))
                if axis == 'x':
                    vertex = np.array((np.clip(radius/0.638, 0, 1), 0, 0))
                elif axis == 'z':
                    vertex = np.array((0, 0, np.clip(radius/0.638, 0, 1)))
                expect = template(vertex, coef, PMT)
                e_mean.append(np.mean(2*a/expect.sum()))
                e_std.append(np.std(2*a/expect.sum()))

            mean = np.array(mean)
            std = np.array(std)
            std = np.array(std)/(mean[0]/2)
            mean = mean/mean[0]*2
            e_mean = np.array(e_mean)
            e_std = np.array(e_std)

            if(axis == 'x'):
                ax1.scatter(np.arange(0.01, 0.65, 0.01)*1000, e_mean, color='k')
                ax1.fill_between(np.arange(0.01, 0.65, 0.01)*1000, e_mean - e_std, e_mean + e_std, color='k', alpha=0.3,
                                label = 'Fit(TrueV)')
                ax1.scatter(np.arange(0.01, 0.65, 0.01)*1000, mean, color='b')
                ax1.fill_between(np.arange(0.01, 0.65, 0.01)*1000, mean - std, mean + std, color='b', alpha=0.3,
                                label = 'Scaled total PE')

            elif axis == 'z':
                ax2.scatter(np.arange(0.01, 0.65, 0.01)*1000, e_mean, color='k')
                ax2.fill_between(np.arange(0.01, 0.65, 0.01)*1000, e_mean - e_std, e_mean + e_std, color='k', alpha=0.3,
                                label = 'Fit(TrueV)')
                ax2.scatter(np.arange(0.01, 0.65, 0.01)*1000, mean, color='b')
                ax2.fill_between(np.arange(0.01, 0.65, 0.01)*1000, mean - std, mean + std, color='b', alpha=0.3,
                                label = 'Scaled total PE')
        for ax in [ax1, ax2]:       
            ax.set(xlim = (0,640), xlabel='Vertex radius/mm', ylabel='Energy/MeV')
            ax.axhline(2, linestyle='dashed', c='k', label='Truth', linewidth=1)
            ax.legend()
            plt.xticks(np.arange(0,650,100), ['%d' % i for i in np.arange(0,650,100)])
        pp.savefig(fig1)
        pp.savefig(fig2)
