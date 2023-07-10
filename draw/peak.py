import tables
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG
from zernike import RZern
from numba import njit

@njit
def polyval(p, x):
    y = np.zeros((p.shape[1], x.shape[0]))
    for i in range(len(p)):
        y = y * x + p[i]
    return y


def radial(coefnorm, rhotab, k, rho):
    return coefnorm[k, np.newaxis] * polyval(rhotab.T[:, k, np.newaxis], rho)

@njit
def angular(m, theta):
    return np.cos(m.reshape(-1, 1) * theta)

plt.figure(dpi=300)
for index, i in enumerate(np.arange(0.26, 0.27, 0.01)):
    h = tables.open_file('/mnt/stage/douwei/JP_1t_paper/concat/shell/%.2f.h5' % i)
    if index == 0:
        data = np.cos(h.root.Concat[:]['theta'])
        Evt = h.root.Concat[:]['EId']
        data0 =  np.cos(h.root.Vertices[:]['theta'])
    else:
        data = np.hstack((data, np.cos(h.root.Concat[:]['theta'])))
        Evt = np.hstack((Evt, h.root.Concat[:]['EId']))
        data0 = np.hstack((data0, np.cos(h.root.Vertices[:]['theta'])))
    h.close()
    
    h1 = tables.open_file('../coeff/Legendre/PE/2/%.2f/30.h5' % i)
    coeff = h1.root.coeff[:]
    h1.close()

    h2 = tables.open_file('../coeff/Zernike/PE/2/30.h5')
    coef_z = h2.root.coeff[:]
    h2.close()

    cart = RZern(30)
    zo = cart.mtab>=0
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta) * i/0.65
    zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], r)
    zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), theta)
    probe = np.matmul((zs_radial * zs_angulars).T, coef_z)

plt.hist(data, bins=np.linspace(-1,1,201), weights=np.ones(data.shape)/Evt.max()*100/30, histtype='step', color='k', label='MC')
plt.plot(np.linspace(-1,1,201), np.exp(LG.legval(np.linspace(-1,1,201), coeff)), c='r', linewidth=1, label='Fit', alpha=0.7)
# plt.plot(np.cos(theta), np.exp(probe), c='b', linewidth=1, label='Zernike', alpha=0.7)

ax = plt.gca()
axins = ax.inset_axes([0.7, 0.05, 0.3, 0.4])
axins.hist(data, bins=np.linspace(-1,1,201), weights=np.ones(data.shape)/Evt.max()*100/30, histtype='step', color='k')
axins.plot(np.linspace(-1,1,201), np.exp(LG.legval(np.linspace(-1,1,201), coeff)), c='r', linewidth=1, label='Fit', alpha=0.7)
# sub region of the original image
x1, x2, y1, y2 = -1, -0.9, 0, 6
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")

plt.xlim(-1, 1)
plt.xlabel(r'$\cos\theta$')
plt.ylabel('Predicted PE')
plt.gca().tick_params(labelsize=18)
plt.legend(fontsize=25)
plt.axhline(3.5, c='r', ls='--', lw=1)
plt.savefig('peak1.pdf')
