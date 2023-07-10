import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG
import tables
import numpy as np


import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages


with PdfPages('Recon.pdf') as pp:
    path = '/mnt/stage/douwei/Upgrade/0.9/recon/point/0.9/'
    data = []
    for radius in np.arange(0.01, 0.65, 0.01):
        with tables.open_file('%s/x/2/%.2f.h5' % (path, radius)) as h:
            index = h.root.ReconIn[:]['Likelihood'] < h.root.ReconOut[:]['Likelihood'] 
            x = h.root.ReconIn[:]['x']
            x[~index] = h.root.ReconOut[:]['x'][~index]
            y = h.root.ReconIn[:]['y']
            y[~index] = h.root.ReconOut[:]['y'][~index]
            z = h.root.ReconIn[:]['z']
            z[~index] = h.root.ReconOut[:]['z'][~index]

            xt = h.root.Truth[:]['x']/1000
            yt = h.root.Truth[:]['y']/1000
            zt = h.root.Truth[:]['z']/1000
        data.append(np.vstack((x, y, z, xt, yt, zt)).T)

    def cb():
        viridis = cm.get_cmap('jet', 256)
        colors = viridis(np.linspace(0, 1, 65536))
        wt = np.array([1, 1, 1, 1])
        colors[:25, :] = wt
        cmp = ListedColormap(colors)
        return cmp
    fig = plt.figure(figsize=(4,4))
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 0])
    data = np.vstack(data)
    ax.hist2d(data[:,0], data[:,3], bins=(np.arange(0.01,0.65,0.01) - 0.005, np.arange(0.01,0.65,0.01) - 0.005), cmap=cb())
    # plt.axis('equal')
    ax.set(xlim = (0,0.64), ylim = (0,0.64), xlabel='$x$/m', ylabel='$x$/m')
    pp.savefig(fig)
    
    path = '/mnt/stage/douwei/JP_1t_paper_bak/recon/point/'
    data = []
    for radius in np.arange(0.01, 0.55, 0.01):
        with tables.open_file('%s/x/2/%.2f.h5' % (path, radius)) as h:
            index = h.root.ReconIn[:]['Likelihood'] < h.root.ReconOut[:]['Likelihood'] 
            x = h.root.ReconIn[:]['x']
            x[~index] = h.root.ReconOut[:]['x'][~index]
            y = h.root.ReconIn[:]['y']
            y[~index] = h.root.ReconOut[:]['y'][~index]
            z = h.root.ReconIn[:]['z']
            z[~index] = h.root.ReconOut[:]['z'][~index]
            xt = h.root.Truth[:]['x']/1000
            yt = h.root.Truth[:]['y']/1000
            zt = h.root.Truth[:]['z']/1000
        data.append(np.vstack((x, y, z, xt, yt, zt)).T)

    for radius in np.arange(0.55, 0.65, 0.01):    
        with tables.open_file('%s/x/2/%.3f.h5' % (path, radius)) as h:
            index = h.root.ReconIn[:]['Likelihood'] < h.root.ReconOut[:]['Likelihood'] 
            x = h.root.ReconIn[:]['x']
            x[~index] = h.root.ReconOut[:]['x'][~index]
            y = h.root.ReconIn[:]['y']
            y[~index] = h.root.ReconOut[:]['y'][~index]
            z = h.root.ReconIn[:]['z']
            z[~index] = h.root.ReconOut[:]['z'][~index]
            xt = h.root.Truth[:]['x']/1000
            yt = h.root.Truth[:]['y']/1000
            zt = h.root.Truth[:]['z']/1000
        data.append(np.vstack((x, y, z, xt, yt, zt)).T)

    fig = plt.figure(figsize=(4,4))
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 0])
    data = np.vstack(data)
    ax.hist2d(data[:,0], data[:,3], bins=(np.arange(0.01,0.65,0.01) - 0.005, np.arange(0.01,0.65,0.01) - 0.005), cmap=cb())
    # plt.axis('equal')
    ax.set(xlim = (0,0.64), ylim = (0,0.64), xlabel='$x$/m', ylabel='$x$/m')
    pp.savefig(fig)
    
    path = '/mnt/stage/douwei/JP_1t_paper_bak/recon/point/'
    data = []
    for radius in np.arange(0.01, 0.55, 0.01):
        with tables.open_file('%s/z/2/%.2f.h5' % (path, radius)) as h:
            index = h.root.ReconIn[:]['Likelihood'] < h.root.ReconOut[:]['Likelihood'] 
            x = h.root.ReconIn[:]['x']
            x[~index] = h.root.ReconOut[:]['x'][~index]
            y = h.root.ReconIn[:]['y']
            y[~index] = h.root.ReconOut[:]['y'][~index]
            z = h.root.ReconIn[:]['z']
            z[~index] = h.root.ReconOut[:]['z'][~index]
            xt = h.root.Truth[:]['x']/1000
            yt = h.root.Truth[:]['y']/1000
            zt = h.root.Truth[:]['z']/1000
        data.append(np.vstack((x, y, z, xt, yt, zt)).T)

    for radius in np.arange(0.55, 0.65, 0.01):    
        with tables.open_file('%s/z/2/%.3f.h5' % (path, radius)) as h:
            index = h.root.ReconIn[:]['Likelihood'] < h.root.ReconOut[:]['Likelihood'] 
            x = h.root.ReconIn[:]['x']
            x[~index] = h.root.ReconOut[:]['x'][~index]
            y = h.root.ReconIn[:]['y']
            y[~index] = h.root.ReconOut[:]['y'][~index]
            z = h.root.ReconIn[:]['z']
            z[~index] = h.root.ReconOut[:]['z'][~index]
            xt = h.root.Truth[:]['x']/1000
            yt = h.root.Truth[:]['y']/1000
            zt = h.root.Truth[:]['z']/1000
        data.append(np.vstack((x, y, z, xt, yt, zt)).T)

    fig = plt.figure(figsize=(4,4))
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 0])
    data = np.vstack(data)
    ax.hist2d(data[:,2], data[:,5], bins=(np.arange(0.01,0.65,0.01) - 0.005, np.arange(0.01,0.65,0.01) - 0.005), cmap=cb())
    # plt.axis('equal')
    ax.set(xlim = (0,0.64), ylim = (0,0.64), xlabel='$z$/m', ylabel='$z$/m')
    pp.savefig(fig)