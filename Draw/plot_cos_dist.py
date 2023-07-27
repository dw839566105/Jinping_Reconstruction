import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
import tables
import numpy as np
import sys
from matplotlib.backends.backend_pdf import PdfPages
import uproot
from scipy.spatial import distance

with PdfPages('cos_dist.pdf') as pp:
    for axis in ['x', 'z']:
        rad = np.arange(0,0.65,0.01)
        Qs = np.empty((len(rad), 30))
        for index, i in enumerate(rad):
            with uproot.open('/mnt/stage/douwei/JP_1t_github/root/point/%s/2/%.2f.root' % (axis, i)) as f:
                data = f['SimTriggerInfo']['PEList/PEList.PMTId'].array(library='np')
            Qs[index] , _ = np.histogram(np.hstack(data), bins=np.arange(31)-0.5, weights=np.full(len(np.hstack(data)),1/len(data)))
        Y = distance.pdist(Qs, 'cosine')
        v= distance.squareform(Y)
        fig = plt.figure(figsize=(5,5))
        ax = plt.gca()
        if axis == 'x':
            CS = ax.contourf(rad*1000, rad*1000, v, levels=30, cmap='gray')
        elif axis == 'z':
            CS = ax.contourf(rad*1000, rad*1000, v, levels=30, cmap='gray')
            ax.axhline(100, color='red', linestyle='dashed')
        ax.tick_params(labelsize=18)
        ax.set_xlabel('Vertex $\mathbf{r}_1$ radius/mm', fontsize=20)
        ax.set_ylabel('Vertex $\mathbf{r}_2$ radius/mm', fontsize=20)
        ax.set_aspect(1)
        cb = fig.colorbar(CS, shrink=0.75, format='%.2f', pad=0.01)
        cb.outline.set_visible(False)
        pp.savefig()
    plt.close()