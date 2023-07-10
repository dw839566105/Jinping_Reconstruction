import numpy as np
import matplotlib.pyplot as plt
import tables
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

with PdfPages('close.pdf') as pp:
    x = []
    y = []
    z = []
    xt = []
    yt = []
    zt = []
    for i in np.arange(1,30):
        with tables.open_file('/mnt/stage/douwei/JP_1t_paper/recon/shell/0.26/2/%02d.h5' % i) as h:
            xt.append(h.root.Truth[:]['x'])
            yt.append(h.root.Truth[:]['y'])
            zt.append(h.root.Truth[:]['z'])
            x.append(h.root.ReconIn[:]['x'])
            y.append(h.root.ReconIn[:]['y'])
            z.append(h.root.ReconIn[:]['z'])
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    xt = np.hstack(xt)
    yt = np.hstack(yt)
    zt = np.hstack(zt)
    v = np.vstack((xt, yt, zt)).T
    PMT = np.loadtxt('../PMT_1t.txt')
    cth = np.einsum('ik, jk->ij', v/1000, PMT) / np.outer(np.linalg.norm(v/1000, axis=1), np.linalg.norm(PMT, axis=1))
    idx = (cth.min(1) < -0.95) & (cth.max(1) > 0.95)
    plt.figure()
    plt.hist(np.sqrt(x**2+y**2+z**2)[idx], bins=100)
    plt.savefig('test.png')
    plt.figure()
    plt.hist(np.sqrt(x**2+y**2+z**2)[~idx], bins=100)
    # plt.hist2d(np.arctan2(y, x)[np.sqrt(x**2+y**2+z**2)>0.55], (z / np.sqrt(x**2 + y**2 + z**2))[np.sqrt(x**2+y**2+z**2)>0.55],bins=(80, 80), norm=LogNorm())
    plt.savefig('test1.png')
    breakpoint()
    
    
    plt.figure()
    plt.hist2d(np.arctan2(y, x), z / np.sqrt(x**2 + y**2 + z**2),
              bins=(80, 80), vmin=5, vmax=80)
    cb = plt.colorbar(pad=0.01)
    cb.outline.set_visible(False)
    PMT_pos = np.loadtxt('/home/douwei/Recon1t/calib/PMT_1t.txt')
    a = np.arctan2(PMT_pos[:,1], PMT_pos[:,0]) + np.pi
    a[a>np.pi] = a[a>np.pi] - np.pi * 2
    plt.scatter(a, -PMT_pos[:,2]/0.83, s=200, marker='o', label='PMT',
               facecolors='none', edgecolors='r',linewidth=2.5)
    plt.gca().tick_params(labelsize=18)
    plt.axhline(-0.30, c='red')
    plt.axvline(1.9, c='red')
    plt.xlabel('$\phi_v$')
    plt.ylabel(r'$\cos\theta_v$')
    pp.savefig()
    
    x = []
    y = []
    z = []
    for i in np.arange(1,30):
        with tables.open_file('/mnt/stage/douwei/JP_1t_paper/recon_close/shell/0.26/2/%02d.h5' % i) as h:
            x.append(h.root.ReconIn[:]['x'])
            y.append(h.root.ReconIn[:]['y'])
            z.append(h.root.ReconIn[:]['z'])

    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    breakpoint()
    plt.figure()
    plt.hist2d(np.arctan2(y, x), z / np.sqrt(x**2 + y**2 + z**2),
              bins=(80, 80), vmin=5, vmax=80)
    cb = plt.colorbar(pad=0.01)
    cb.outline.set_visible(False)
    
    PMT_pos = np.loadtxt('/home/douwei/Recon1t/calib/PMT_1t.txt')
    a = np.arctan2(PMT_pos[:,1], PMT_pos[:,0]) + np.pi
    a[a>np.pi] = a[a>np.pi] - np.pi * 2
    index = np.arange(30) != 11
    plt.scatter(a[index], -PMT_pos[index,2]/0.83, s=200, marker='o', label='PMT',
               facecolors='none', edgecolors='r',linewidth=2.5)
    plt.scatter(a[~index], -PMT_pos[~index,2]/0.83, s=200, marker='^', label='masked',
               facecolors='none', edgecolors='r',linewidth=2.5)
    plt.gca().tick_params(labelsize=18)
    plt.axhline(-0.31, c='red')
    plt.axvline(1.9, c='red')
    plt.xlabel('$\phi_v$')
    plt.ylabel(r'$\cos\theta_v$')
    plt.legend()
    plt.figure()
    plt.hist(np.sqrt(x**2+y**2+z**2), bins=100)
    plt.savefig('test2.png')
    pp.savefig()