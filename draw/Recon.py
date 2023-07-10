import numpy as np
import tables
import pandas as pd
from polynomial import *
from zernike import RZern
from numpy.polynomial import legendre as LG
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
from tqdm import tqdm
import uproot
import awkward as ak

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

with PdfPages('Recon.pdf') as pp:
    for prefix in ['/mnt/stage/douwei/Upgrade/0.90/recon/point', '/mnt/stage/douwei/JP_1t_paper/recon/point']:
        for axis in ['x','z']:
            path = '%s/%s/2' % (prefix, axis)
            data = []
            for radius in np.arange(0.01, 0.65, 0.01):
                with tables.open_file('%s/%.2f.h5' % (path, radius)) as h:
                    v = tract(h)
                    h.close()
                data.append(np.vstack((v)).T)
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(1,1,1)
            for idx, i in enumerate(data):
                idx = idx + 1
                if axis == 'x':
                    col = int(0)
                elif axis == 'z':
                    col = int(2)
                ax.boxplot(i[:,col], positions=(idx/100,), widths=(0.005), flierprops=dict(markerfacecolor='k', marker='.', markersize=1, alpha=0.5))
            ax.plot([0, 0.65], [0, 0.65], 'k')
            ax.set(xlim = (0,0.64), ylim = (0,0.64), xlabel = '$%s$ by MC/m' % axis, ylabel = '$%s$ by Recon/m' % axis)
            plt.xticks(np.arange(0,0.65,0.1), ['%.1f' % i for i in np.arange(0,0.65,0.1)])
            plt.yticks(np.arange(0,0.65,0.1), ['%.1f' % i for i in np.arange(0,0.65,0.1)])
            pp.savefig(fig)

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(1,1,1)
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1,1,1)
            for idx, i in enumerate(data):
                idx = idx + 1
                if axis == 'x':
                    p1 = ax1.scatter(idx/100, i[:,0].mean() - idx/100, label='x', color='r')
                else:
                    p1 = ax1.scatter(idx/100, i[:,0].mean(), label='x', color='r')
                p2 = ax1.scatter(idx/100, i[:,1].mean(), label='y', color='b')
                if axis == 'z':
                    p3 = ax1.scatter(idx/100, i[:,2].mean() - idx/100, label='z', color='g')
                else:
                    p3 = ax1.scatter(idx/100, i[:,2].mean(), label='z', color='g')
                q1 = ax2.scatter(idx/100, i[:,0].std(), label='x', color='r')
                q2 = ax2.scatter(idx/100, i[:,1].std(), label='y', color='g')
                q3 = ax2.scatter(idx/100, i[:,2].std(), label='z', color='b')
            ax1.legend([p1, p2, p3], ['x', 'y', 'z'])
            ax1.set(xlabel = 'Radius/m', ylabel = 'Bias/m')
            ax2.legend([q1, q2, q3], ['x', 'y', 'z'])
            ax2.set(xlabel = 'Radius/m', ylabel = 'Std/m')
            pp.savefig(fig1)
            pp.savefig(fig2)

    for prefix in ['/mnt/stage/douwei/Upgrade/0.90/root/point', '/mnt/stage/douwei/JP_1t_paper/root/point']:
        
        if prefix == '/mnt/stage/douwei/Upgrade/0.90/root/point':
            PMT = np.loadtxt('/home/douwei/UpgradeJP/DetectorStructure/1t_0.90/PMT.txt')
            nPMT = int(121)
        else:
            PMT = np.loadtxt('/home/douwei/ReconJP/PMT.txt')
            nPMT = int(31)
            
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

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(1,1,1)
            for idx, i in enumerate(data):
                idx = idx + 1
                if axis == 'x':
                    col = int(0)
                elif axis == 'z':
                    col = int(2)
                pdata = i[:,col]
                pdata = pdata[~np.isnan(pdata)]
                ax.boxplot(pdata, positions=(idx/100,), widths=(0.005), flierprops=dict(markerfacecolor='k', marker='.', markersize=1, alpha=0.5))
            ax.plot([0, 0.65], [0, 0.65], 'k')
            ax.set(xlim = (0,0.64), ylim = (0,0.64), xlabel = '$%s$ by MC/m' % axis, ylabel = '$%s$ by Recon/m' % axis)
            plt.xticks(np.arange(0,0.65,0.1), ['%.1f' % i for i in np.arange(0,0.65,0.1)])
            plt.yticks(np.arange(0,0.65,0.1), ['%.1f' % i for i in np.arange(0,0.65,0.1)])
            pp.savefig(fig)