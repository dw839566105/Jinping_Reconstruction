import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG
import tables
import numpy as np

import sys
output = sys.argv[1]

from tqdm import tqdm

def load_data(data, recon=True):
    E = data['E']
    x = data['x']
    y = data['y']
    z = data['z']
    if recon:
        L = data['Likelihood']
        s = data['success']
        return E, x, y, z, L, s
    else:
        return E, x, y, z

def main(path, radius):
    try:
        h = tables.open_file(path + '%.2f.h5' % radius, 'r')
    except:
        h = tables.open_file(path + '%.3f.h5' % radius, 'r')
    reconin = h.root.ReconIn[:]
    reconout = h.root.ReconOut[:]
    truth = h.root.Truth[:]
    E1, x1, y1, z1, L1, s1 = load_data(reconin)
    E2, x2, y2, z2, L2, s2 = load_data(reconout)
    Et, xt, yt, zt = load_data(truth, False)
    
    h.close()
    
    data = np.zeros((np.size(x1),3))

    index = L1 < L2
    data[index, 0] = x1[index]
    data[index, 1] = y1[index]
    data[index, 2] = z1[index]

    data[~index, 0] = x2[~index]
    data[~index, 1] = y2[~index]
    data[~index, 2] = z2[~index]

    idx = s1*s2 != 0 
    x = data[(s1 * s2)!=0, 0]
    y = data[(s1 * s2)!=0, 1]
    z = data[(s1 * s2)!=0, 2]
    xt = xt[idx]
    yt = yt[idx]
    zt = zt[idx]
    # r = np.sqrt(x**2 + y**2 + z**2)
    # index1 = (r<0.60) & (r>0.01) & (~np.isnan(r))
    return np.std(x), np.std(y), np.std(z),\
        np.mean(x - xt/1000),np.mean(y - yt/1000),np.mean(z - zt/1000)
                                          
A = []
ra = np.arange(0.01, 0.65, 0.01)
for i in tqdm(ra):
    # A.append(main('/mnt/stage/douwei/JP_1t_paper_bak/recon/point/x/2/', i))
    A.append(main('/mnt/stage/douwei/Upgrade/0.85/recon/point/0.85/x/2/', i))
A = np.array(A)
#plt.subplots(2,2, sharex=True)
plt.subplot(2,2,1)
plt.plot(ra, A[:,0],'r.', alpha=0.8, label=r'$x$')
plt.plot(ra, A[:,1],'g.', alpha=0.8, label=r'$y$')
plt.plot(ra, A[:,2],'b.', alpha=0.8, label=r'$z$')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
#plt.xlabel('Radius/\si{m}', fontsize=20)
plt.ylabel('Std/m', fontsize=20)
plt.legend(fontsize=15)
plt.title(r'$x$ resolution', fontsize=20)

plt.subplot(2,2,3)
plt.plot(ra, A[:,3],'r.', alpha=0.8, label=r'$x$')
plt.plot(ra, A[:,4],'g.', alpha=0.8, label=r'$y$')
plt.plot(ra, A[:,5],'b.', alpha=0.8, label=r'$z$')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
plt.xlabel('Radius/m', fontsize=20)
plt.ylabel('Bias/m', fontsize=20)
plt.legend(fontsize=15)
plt.title(r'$x$ bias', fontsize=20)
     
A = []
ra = np.arange(0.01, 0.65, 0.01)
for i in tqdm(ra):
    # A.append(main('/mnt/stage/douwei/JP_1t_paper_bak/recon/point/x/2/', i))
    # A.append(main('/mnt/stage/douwei/Upgrade/0.9/recon/point/0.9/x/2/', i))
    A.append(main('/mnt/stage/douwei/JP_1t_paper_bak/recon/point/x/2/', i))
A = np.array(A)
plt.subplot(2,2,2)
plt.plot(ra, A[:,0],'r.', alpha=0.8, label=r'$x$')
plt.plot(ra, A[:,1],'g.', alpha=0.8, label=r'$y$')
plt.plot(ra, A[:,2],'b.', alpha=0.8, label=r'$z$')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
plt.ylabel('Std/m', fontsize=20)
plt.legend(fontsize=15)
plt.title(r'$z$ resolution', fontsize=20)

plt.subplot(2,2,4)
plt.plot(ra, A[:,3],'r.', alpha=0.8, label=r'$x$')
plt.plot(ra, A[:,4],'g.', alpha=0.8, label=r'$y$')
plt.plot(ra, A[:,5],'b.', alpha=0.8, label=r'$z$')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
plt.xlabel('Radius/m', fontsize=20)
plt.ylabel('Bias/m', fontsize=20)
plt.legend(fontsize=15)
plt.title(r'$z$ bias', fontsize=20)

plt.gcf().align_ylabels()
plt.gcf().align_xlabels()
plt.tight_layout()
plt.savefig(output)
plt.close()
