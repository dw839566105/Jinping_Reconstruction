import matplotlib as mpl
import seaborn as sns
mpl.use('pdf')
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG

import tables
import numpy as np

import sys

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import LogNorm

output = sys.argv[1]

plt.rcParams['lines.markersize'] = 5

PMT_pos = np.loadtxt('/home/douwei/ReconJP/PMT_1t.txt')

viridis = cm.get_cmap('jet', 256)
newcolors = viridis(np.linspace(0, 1, 65536))
wt = np.array([1, 1, 1, 1])
newcolors[:25, :] = wt
newcmp = ListedColormap(newcolors)

def Legendre_coeff(PMT_pos_rep, vertex, cut):
    '''
    # calulate the Legendre value of transformed X
    # input: PMT_pos: PMT No * 3
          vertex: 'v' 
          cut: cut off of Legendre polynomial
    # output: x: as 'X' at the beginnig    
    
    '''
    size = np.size(PMT_pos_rep[:,0])
    # oh, it will use norm in future version
    
    if(np.sum(vertex**2) > 1e-6):
        cos_theta = np.sum(vertex*PMT_pos_rep,axis=1)\
            /np.sqrt(np.sum(vertex**2, axis=1)*np.sum(PMT_pos_rep**2,axis=1))
    else:
        # make r=0 as a boundry, it should be improved
        cos_theta = np.ones(size)

    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)

    # print(PMT_pos_rep.shape, x.shape, cos_theta.shape)
    return x, cos_theta

plt.gca().set_facecolor('w')
mpl.rcParams['axes.edgecolor'] = 'k'
h = tables.open_file('../coeff/Legendre/Time/0.620/20.h5')
z = np.linspace(-1,1,100)
k = LG.legval(z, h.root.coeff[:])
h.close()
p1 = plt.plot(z, k, 'r-')

h1 = tables.open_file('/mnt/stage/douwei/JP_1t_paper/concat/shell/0.620.h5')
theta = h1.root.Concat[:]['theta']
t = h1.root.Concat[:]['t']
h1.close()
p0 = plt.hist2d(np.cos(theta), t, bins=(np.linspace(-1,1,80), np.linspace(0,200,80)), cmap=newcmp, norm = LogNorm())
# plt.gca().tick_params(axis = 'both', which = 'major', labelsize = 15)
plt.xlabel(r'$\cos\theta$')
plt.ylabel('Hit time/ns')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.legend(handles=(p1[0],), labels=(['qt regression']), facecolor='w')
plt.tight_layout()
plt.savefig(output)
plt.close()