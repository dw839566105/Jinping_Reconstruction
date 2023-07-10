import numpy as np
import tables
import matplotlib.pyplot as plt
import pandas as pd
from zernike import RZern
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from polynomial import *
from argparse import RawTextHelpFormatter
import argparse
# plt.rc('text', usetex=False)
parser = argparse.ArgumentParser(description='Process template construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('--pe', dest='pe', metavar='PE[*.h5]', type=str,
                            help='The pe coefficient [.h5] to read')

parser.add_argument('--time', dest='time', metavar='Time[*.h5]', type=str,
                            help='The time coefficient [*.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='Coeff[*.h5]', type=str,
                            help='The output file [*.h5] to save')
args = parser.parse_args()

r_max = 638
neighborhood_r = 0.05

vertices = np.array(
    [
        (0.99, 0, '0'),
        (0.99,np.pi/4, r'$\frac{\pi}{4}$'),
        (0.99,np.pi, r'$\pi$'),
        (0.5,np.pi, r'$\pi$'),
        (0, 0, '0'),
    ],
    dtype=[
        ("r", np.float64),
        ("theta", np.float64),
        ("theta_text", object),
    ],
)


def loadh5(filename):
    h = tables.open_file(filename)
    coef_ = h.root.coeff[:]
    coef_type = h.root.coeff.attrs.type
    coef_order = h.root.coeff.attrs.order
    h.close()
    return coef_, coef_type, coef_order

coef_PE, coef_type1, coef_order1 = loadh5(args.pe)
coef_Time, coef_type2, coef_order2 = loadh5(args.time)

thetas = np.linspace(0, 2 * np.pi, 201)
rs = np.linspace(0, 1, 101)

def calc_probe(ddx, ddy, coef, coef_type, order, pad=False):
    if pad:
        rho, theta = ddx, ddy
    else:
        rho, theta = np.meshgrid(ddx[:-1], ddy[:-1])
    rhof = rho.flatten()
    theta = theta.flatten()
    if(coef_type=='Zernike'):
        cart = RZern(order)
        zo = cart.mtab>=0
        zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], rhof)
        zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), theta)
        probe = np.matmul((zs_radial * zs_angulars).T, coef)
    elif(coef_type=='Legendre'):
        cut = len(coef)
        t_basis = legval_raw(np.cos(theta), np.eye(cut).reshape((cut,cut,1))).T
        # r_basis = legval_raw(rhof, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
        r_basis = legval_raw(rhof, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
        probe = (t_basis*r_basis).sum(-1)

    elif(coef_type=='db_Legendre'):

        o1, o2 = 10, 10
        X1 = legval_raw(rhof, np.eye(2*o1).reshape((2*o1, 2*o1, 1))).T
        X1 = X1[:,::2]
        X2 = legval_raw(np.cos(theta), np.eye(o2).reshape((o2, o2, 1))).T
        
        X = np.empty((len(X1), o1*o2))
        for i in range(o1):
            for j in range(o2):
                X[:,i*o2 + j] = X1[:,i] * X2[:,j]
        
        probe = np.dot(X, coef)
    return probe.reshape(rho.shape)

def concat():
    for index, i in enumerate(np.arange(1,30)):
        h = tables.open_file(f'/mnt/stage/douwei/JP_1t_paper/concat/ball/{i:02d}.h5')
        # h = tables.open_file(f'/mnt/stage/douwei/JP_1t_paper_bak/concat/ball/2/{i:02d}.h5')
        if index == 0:
            df_nonhit = pd.DataFrame(h.root.Vertices[:])
            df_hit = pd.DataFrame(h.root.Concat[:])
        else:
            df_nonhit = pd.concat([df_nonhit, pd.DataFrame(h.root.Vertices[:])], 
                ignore_index=True)
            df_hit = pd.concat([df_hit, pd.DataFrame(h.root.Concat[:])], 
                ignore_index=True)
        h.close()
    return df_hit, df_nonhit

def calc_real(ddx, ddy):
    _hit = df_hit.copy()
    _hit['theta'] = 2 * np.pi - _hit['theta']
    _nonhit = df_nonhit.copy()
    _nonhit['theta'] = 2 * np.pi - _nonhit['theta']
    df_hit_ = pd.concat([df_hit, _hit])
    df_nonhit_ = pd.concat([df_nonhit, _nonhit])
    H, _, _ = np.histogram2d(df_hit_['r']/r_max, df_hit_['theta'], bins = (ddx, ddy))
    H_prior, _, _ = np.histogram2d(df_nonhit_['r']/r_max, df_nonhit_['theta'], bins = (ddx, ddy))
    return H, H_prior

def neightbor(r, theta, r0, theta0):
    index = (r ** 2 + r0 ** 2 - 2 * r * r0* np.cos(theta - theta0) <= neighborhood_r ** 2)
    return index

def time_hist(r0, theta0, df_nonhit, df_hit):
    index_hit = neightbor(df_hit['r']/r_max, df_hit['theta'], r0, theta0)
    index_nonhit = neightbor(df_nonhit['r']/r_max, df_nonhit['theta'], r0, theta0)
    return df_nonhit[index_nonhit], df_hit[index_hit]

def quantile(xx, tau, ts, expect):
    return tau*(1-tau)/ts * np.exp(-1/ts *((1 - tau)* (expect - xx) * (expect > xx) + tau* (xx- expect) * (xx >= expect)))

def time_plot(df1, df2, ax):
    expect_PE = calc_probe(df1['r'].values/r_max, df1['theta'].values, coef_PE, coef_type1, coef_order1, pad=True)
    expect = calc_probe(df2['r'].values/r_max, df2['theta'].values, coef_Time, coef_type2, coef_order2, pad=True)
    xx = np.arange(0,200,1)
    time_pdf = quantile(xx, tau=0.1, ts=3, expect = expect[0])
    ax.plot(np.mean(np.exp(expect_PE)[:,np.newaxis] * time_pdf, axis=0), label='fit')
    ax.hist(df2['t'], bins = xx, weights = 1/len(df1) * np.ones(len(df2)), label = 'truth')
    ax.set_xlabel('Time/ns')
    # ax.set_ylabel('Poisson flux')
    ax.set_ylabel(r'$\lambda(t)$')

df_hit, df_nonhit = concat()

probe = calc_probe(rs, thetas, coef_PE, coef_type1, coef_order1).T
probe_time = calc_probe(rs, thetas, coef_Time, coef_type2, coef_order2).T

H, H_prior = calc_real(rs, thetas)

with PdfPages(args.output) as pdf:
    X, Y = np.meshgrid(thetas, rs)

    for data, strs, ctitle in zip([np.exp(probe), probe_time, H/H_prior, np.exp(probe)/(H/H_prior)], ['PE', 'Time', 'Real', 'Quotient'], ['PE', 'time/ns', 'PE', 'ratio']):
        fig = plt.figure(dpi=200)
        ax = plt.subplot(1, 1, 1, projection="polar", theta_offset=np.pi / 2)
        if strs == 'Time':
            im = ax.pcolormesh(X, Y, data, cmap='binary')
        else:
            im = ax.pcolormesh(X, Y, data, norm = LogNorm(), cmap='gray')
        ax.set_xticks(np.linspace(0,2 * np.pi,9)[:-1], [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
        ax.set_rlabel_position(180+22.5)
        ax.tick_params(labelsize=20)
        cb = fig.colorbar(im, pad=0.08)
        cb.ax.set_title(ctitle)
        # if strs == 'PE':
        #    ax.plot(np.linspace(0,np.pi, 100), np.ones(100)*0.26/0.65, c='k', ls='--', lw=2)
        #    ax.plot(np.ones(100)*np.pi/6*5, np.linspace(0,1,100), c='k', ls='dotted', lw=2)
        #ax.set_title(f'Pie of {strs}')
        pdf.savefig(fig)
        plt.close()

    for v in vertices:
        fig = plt.figure(dpi=200)
        ax = plt.gca()
        ax.tick_params(labelsize=20)
        df1, df2 = time_hist(v['r'], v['theta'], df_nonhit, df_hit)
        time_plot(df1, df2, ax)
        ax.set_title(r'$r$ = %.2f and $\theta$ = %s' % (v['r'], v['theta_text']))
        ax.legend()
        pdf.savefig(fig)
        plt.close()