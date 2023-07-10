import matplotlib.pyplot as plt
import tables
import numpy as np
from numba import njit
from matplotlib.backends.backend_pdf import PdfPages
from zernike import RZern

@njit
def legval(x, n):
    res = np.zeros((n,) + x.shape)
    res[0] = 1
    res[1] = x
    for i in range(2, n):
        res[i] = ((2 * i - 1) * x * res[i - 1] - (i - 1) * res[i - 2]) / i
    return res


@njit
def legval_raw(x, c):
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x


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


N = 300
r = np.linspace(0, 0.99, N)
theta = np.linspace(0, np.pi*2, N)
rr, tt = np.meshgrid(r, theta, sparse=False)
rrep = rr.flatten()
trep = tt.flatten()
cart = RZern(30)
zo = cart.mtab>=0
zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], rrep)
zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), trep)

with PdfPages('slice.pdf') as pp:
    fig = plt.figure(tight_layout=True)
    for i, c in zip(['0.85', '0.9', '1.0'], ['r', 'g', 'b']):
        with tables.open_file('/mnt/stage/douwei/Upgrade_bak/%s/coeff/Zernike/PE/%s/2/30.h5' % (i, i)) as h:
            coeff_tmp = h.root.coeff[:]
            probe = np.exp(np.matmul((zs_radial * zs_angulars).T, coeff_tmp)).reshape(rr.shape)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,1))
            r = np.linspace(0, 0.65, N)
            plt.plot(1000*r, probe[int(150*5/6)], c=c, label = '$r_\mathrm{PMT} = %d$ mm' % (eval(i)*1000))
            data = probe[int(150*5/6)]
            # print(data[-50:].max())
            idx1 = np.where(data == data[-50:].max())[0][0]
            plt.axhline(data[idx1], color=c, linewidth=1, linestyle='--')
            idx2 = np.where(np.abs(data[:-50] - data[idx1]) == np.min(np.abs(data[:-50] - data[idx1])))[0][0]
            #plt.fill_between([1000*r[idx1], 1000*r[idx2]], [0.8,0.8],[5.5, 5.5], color=c, linewidth=1, linestyle='--', alpha=0.1)
            # plt.axvline(r[idx1], color=c, linewidth=1, linestyle='--')
            #plt.axvline(r[idx2], color=c, linewidth=1, linestyle='--')
            # print(data[idx])
            # plt.fill_between([0.5,0.6], [0.8,0.8], [5.5, 5.5], color='c')
    # plt.yscale('log')
    plt.ylim(0.8, 5.5)
    plt.gca().tick_params(labelsize=18)
    plt.legend(fontsize=25)
    plt.xlabel('Vertex radius/mm')
    plt.ylabel('Predicted PE')
    pp.savefig(fig)

    fig = plt.figure(tight_layout=True)
    for i in ['0.85', '0.9', '1.0']:
        with tables.open_file('/mnt/stage/douwei/Upgrade_bak/%s/coeff/Zernike/PE/%s/2/30.h5' % (i, i)) as h:
            coeff_tmp = h.root.coeff[:]
            probe = np.exp(np.matmul((zs_radial * zs_angulars).T, coeff_tmp)).reshape(rr.shape)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,1))
            plt.plot(1000*np.linspace(0, 0.65, N), probe[int(150*2/6)], label = '%d' % (eval(i)*1000))
    # plt.yscale('log')
    plt.legend()
    plt.xlabel('Vertex radius/mm')
    plt.ylabel('Expected PE')
    pp.savefig(fig)
