import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
import tables
import numpy as np
import sys
output = sys.argv[1]

data = []
for i in np.arange(0,0.66,0.01):
    h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_xQ.h5' %i)
    Q = h.root.PETruthData[:]['Q']
    data.append(np.mean(np.reshape(Q, (-1,30), order='C'), axis=0))
    h.close()
fig, axes = plt.subplots(nrows=1, ncols=2)

from scipy.spatial import distance
Y = distance.pdist(np.array(data), 'cosine')
v= distance.squareform(Y)
x = np.arange(0,0.66,0.01)
p0 = axes[0].contourf(x/0.65,x/0.65,v,levels=30, vmin=0, vmax = 0.55, cmap='gray')
axes[0].set_xticks(np.linspace(0,1,5))
axes[0].set_yticks(np.linspace(0,1,5))
axes[0].tick_params(axis = 'both', which = 'major')
axes[0].set_xlabel('Relative radius')
axes[0].set_ylabel('Relative radius')  
axes[0].set_title('Cosine distance of $x$')
axes[0].set_aspect(1)
# plt.colorbar()
data = []
for i in np.arange(0,0.66,0.01):
    h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_zQ.h5' %i)
    Q = h.root.PETruthData[:]['Q']
    data.append(np.mean(np.reshape(Q, (-1,30), order='C'), axis=0))
    h.close()
from scipy.spatial import distance
Y = distance.pdist(np.array(data), 'cosine')
v= distance.squareform(Y)
x = np.arange(0,0.66,0.01)
p1 = axes[1].contourf(x/0.65,x/0.65,v,levels=30, vmin=0, vmax = 0.55, cmap='gray')
#axes[1].locator_params(nbins=4, axis='x')
axes[1].set_xticks(np.linspace(0,1,5))
axes[1].set_yticks(np.linspace(0,1,5))
axes[1].tick_params(axis = 'both', which = 'major')
axes[1].set_xlabel('Relative radius')
axes[1].set_yticks([])
axes[1].set_title('Cosine distance of $z$')
axes[1].set_aspect(1)
axes[1].set_xlim([0,1])
axes[1].set_ylim([0,1])
fig.subplots_adjust(right=0.9,wspace=0.15)
cbar = fig.add_axes([0.92, 0.25, 0.01, 0.5])
cb = fig.colorbar(p0, cax=cbar, ticks=np.arange(0,0.6,0.1), shrink=0.9)
#cb.ax.spines['bottom'].set_color('w')
#cb.ax.spines['top'].set_color('w') 
#cb.ax.spines['right'].set_color('w')
#cb.ax.spines['left'].set_color('w')
cb.ax.tick_params(labelsize=12)
cb.outline.set_visible(False)
#plt.tight_layout()

plt.savefig(output)
plt.close()