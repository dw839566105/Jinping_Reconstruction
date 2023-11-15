import numpy as np
import tables
import pandas as pd
from polynomial import *
from zernike import RZern
from numpy.polynomial import legendre as LG
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from tqdm import tqdm
import uproot
import awkward as ak

def tract(h):
    index = (h.root.ReconIn[:]['Likelihood'] < h.root.ReconOut[:]['Likelihood'])
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

with PdfPages('Recon_h.pdf') as pp:
    prefix = '/mnt/stage/douwei/JP_1t_github/recon/point/'
    for axis in ['x','z']:
        path = '%s/%s/2' % (prefix, axis)
        data = []
        for radius in np.arange(0.01, 0.65, 0.01):
            with tables.open_file('%s/%.2f.h5' % (path, radius)) as h:
                v = tract(h)
            if axis == 'x':
                col = int(0)
            elif axis == 'z':
                col = int(2)
            h, _ = np.histogram(np.vstack((v)).T[:, col], bins=np.linspace(-0.005, 0.645, 66))
            data.append(h)
        data = np.vstack(data)
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(1,1,1)
        m = ax.contourf(np.arange(10,650,10), np.arange(0,650,10), data.T, cmap='Greys', norm=LogNorm(), vmax=1000)
        ax.plot([0, 650], [0, 650], 'r')
        ax.set(xlim = (0, 650), ylim = (0, 650))
        ax.set_xlabel('True $%s$/mm' % axis, fontsize=18)
        ax.set_ylabel('Recon $%s$/mm' % axis, fontsize=18)
        plt.xticks(np.arange(0, 650, 200), ['%d' % i for i in np.arange(0,650,200)], fontsize=16)
        plt.yticks(np.arange(0, 650, 200), ['%d' % i for i in np.arange(0,650,200)], fontsize=16)
        
        pp.savefig(fig)

    prefix = '/mnt/stage/douwei/JP_1t_github/root/point/'
    PMT = np.loadtxt('./PMT.txt')
    nPMT = int(30 + 1)
        
    for axis in ['x','z']:
        path = '%s/%s/2' % (prefix, axis)
        data = []
        
        for radius in tqdm(np.arange(0.01, 0.65, 0.01)):
            with uproot.open('%s/%.2f.root' % (path, radius)) as f:
                d = f['SimTriggerInfo']
                pid = d['PEList.PMTId'].array()
                a = [len(i) for i in pid]
                xt = ak.to_numpy(d['truthList.x'].array())
                yt = ak.to_numpy(d['truthList.y'].array())
                zt = ak.to_numpy(d['truthList.z'].array())
                PMTId = ak.to_numpy(ak.flatten(pid))
                SegmentId = ak.to_numpy(ak.flatten(d['truthList.SegmentId'].array()))
                H, _, _ = np.histogram2d(np.repeat(SegmentId, np.array(a), axis=0), PMTId, bins=(np.arange(SegmentId[-1]+2), np.arange(nPMT)))
                point = 1.5*np.matmul(H, PMT)/H.sum(-1)[:, np.newaxis]
                try:
                    data.append(np.hstack((point, np.hstack((xt, yt, zt)))))
                except:
                    data.append(np.hstack((point, 
                                            (np.ones(len(point))*xt[0])[:,np.newaxis],
                                            (np.ones(len(point))*yt[0])[:,np.newaxis],
                                            (np.ones(len(point))*zt[0])[:,np.newaxis])))

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(1,1,1)
        Hdata = []
        for idx, i in enumerate(data):
            idx = idx + 1
            if axis == 'x':
                col = int(0)
            elif axis == 'z':
                col = int(2)
            pdata = i[:,col]
            pdata = pdata[~np.isnan(pdata)]
            h, _ = np.histogram(pdata.T, bins=np.linspace(-0.005, 0.645, 66))
            Hdata.append(h)
        
        Hdata = np.vstack(Hdata)
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(1,1,1)
        m = ax.contourf(np.arange(10, 650, 10), np.arange(0, 650, 10), Hdata.T, norm=LogNorm(), cmap='Greys', vmax=1000)
        ax.plot([0, 650], [0, 650], 'r')
        ax.set(xlim = (0, 650), ylim = (0, 650))
        ax.set_xlabel('True $%s$/mm' % axis, fontsize=18)
        ax.set_ylabel('Recon $%s$/mm' % axis, fontsize=18)
        plt.xticks(np.arange(0, 650, 200), ['%d' % i for i in np.arange(0,650,200)], fontsize=16)
        plt.yticks(np.arange(0, 650, 200), ['%d' % i for i in np.arange(0,650,200)], fontsize=16)
        pp.savefig(fig)