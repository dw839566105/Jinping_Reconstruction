import tables
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG
from zernike import RZern
from numba import njit

PMT = np.loadtxt('/home/douwei/ReconJP/PMT.txt')
theta = PMT[:,2] / np.linalg.norm(PMT, axis=1)

plt.figure(dpi=150)
for index, i in enumerate([0.60, 0.10]):
    if np.isin(index, [0]):
        with tables.open_file('/mnt/stage/douwei/JP_1t_paper/concat/shell/%.3f.h5' % i) as h:
            data = np.cos(h.root.Concat[:]['theta'])
            Evt = h.root.Concat[:]['EId']
            data0 =  np.cos(h.root.Vertices[:]['theta'])
            
        with tables.open_file('../coeff/Legendre/PE/2/%.3f/30.h5' % i) as h1:
            coeff = h1.root.coeff[:]
    else:
        with tables.open_file('/mnt/stage/douwei/JP_1t_paper/concat/shell/%.2f.h5' % i) as h:
            data = np.hstack((data, np.cos(h.root.Concat[:]['theta'])))
            Evt = np.hstack((Evt, h.root.Concat[:]['EId']))
            data0 = np.hstack((data0, np.cos(h.root.Vertices[:]['theta'])))
    
        with tables.open_file('../coeff/Legendre/PE/2/%.2f/30.h5' % i) as h1:
            coeff = h1.root.coeff[:]
    prd = np.exp(LG.legval(theta, coeff))
    
    # print(np.arccos(theta)/np.pi)
    plt.plot(np.linspace(-1,1,201), np.exp(LG.legval(np.linspace(-1,1,201), coeff)) / prd.sum(), linewidth=2, 
             label='$r=$ %d mm' % (i*1000))
    plt.scatter(theta, prd / prd.sum(), label='', c='k')
# plt.scatter(theta, prd / prd.sum(), label='PMTs at outlet direction', c='k')
plt.scatter(theta, prd / prd.sum(), label='PMTs', c='k')
# plt.vlines(theta, ymin=0, ymax=1.5, color='k', ls='--', label=r'PMT sampled $\cos\theta$ on $z$ axis', alpha=0.3)
plt.semilogy()
plt.xlabel(r'$\cos\theta$', fontsize=30)
plt.ylabel(r'Scaled predicted PE', fontsize=30)
plt.gca().tick_params(axis = 'both', which = 'major', labelsize = 25)
plt.xlim(-1, 1)
plt.ylim(0.001, 1.5)
lb = plt.legend(frameon=True, fontsize=25)
for lh in lb.legendHandles: 
    lh.set_alpha(1)
plt.savefig('predict.pdf')