import sys
import pub
import numpy as np

import tables, h5py
from argparse import RawTextHelpFormatter
import argparse, textwrap
import time
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# global:
PMTPos = pub.ReadJPPMT()

def main_Calib(filename, output, mode, order, offset, qt):

    print('begin reading file', flush=True)
    if(offset):
        off = pub.LoadBase(offset)
    else:
        off = np.zeros_like(PMTPos[:,0])

    tmp = time.time()
    ## sphere
    for i in np.arange(2,7):
        if i == 2:
            with h5py.File(filename + '%d.h5' % i, "r") as ipt:
                h_hit = ipt['Concat'][()]
                h_nonhit = ipt['Vertices'][()]
        else:
            with h5py.File(filename + '%d.h5' % i, "r") as ipt:
                data = ipt['Concat'][()]
                data['EId'] += np.max(h_hit['EId'])
                h_hit = np.hstack((h_hit, data))
                h_nonhit = np.hstack((h_nonhit, ipt['Vertices'][()]))
    ## shell
    '''
    for index, i in enumerate(np.arange(0.01, 0.65, 0.01)):
        if index == 0:
            with h5py.File(filename + '%.2f.h5' % i, "r") as ipt:
                h_hit = ipt['Concat'][()]
                h_nonhit = ipt['Vertices'][()]
        else:
            with h5py.File(filename + '%.2f.h5' % i, "r") as ipt:
                data = ipt['Concat'][()]
                data['EId'] += np.max(h_hit['EId'])
                h_hit = np.hstack((h_hit, data))
                h_nonhit = np.hstack((h_nonhit, ipt['Vertices'][()]))
    '''
    '''
    with h5py.File(filename + '%.2f.h5' % 0.40, "r") as ipt:
        h_hit = ipt['Concat'][()]
        h_nonhit = ipt['Vertices'][()]
    '''
    EventId = h_hit['EId']
    ChannelId = h_hit['CId']
    PMTNo = np.size(PMTPos[:,0])
    if mode == 'PE':
        y, _, _ = np.histogram2d(EventId, ChannelId,
            bins = (np.arange(np.min(EventId), np.max(EventId)+2)-0.5, np.arange(31) - 0.5))
        y = y.flatten()
        rho = h_nonhit['r']/645.
        theta = h_nonhit['theta']
    elif mode == 'time':
        y = h_hit['t']
        rho = h_hit['r']/645.
        theta = h_hit['theta']
    
    order1 = 30
    order2 = 30
    # X, cos_theta = pub.LegendreCoeff(PMTPosRep, vertex, order, Legendre=True)
    X0 = pub.legval(np.cos(theta), np.eye(order1*2).reshape(order1*2, order1*2, 1)).T
    X1 = pub.legval(rho, np.eye(order2*2).reshape(order2*2, order2*2, 1)).T
    LL = np.zeros((X1.shape[0], order1*order2))
    for x1 in np.arange(order1):
        if not x1 % 5:
            print(x1)
        for x2 in np.arange(0,2*order2,2):
            LL[:, x1*order2+np.int(x2/2)] = X0[:, x1]*X1[:, x2]
    X = LL
    '''
    order = 30
    X = pub.legval(np.cos(theta), np.eye(order*2).reshape(order*2, order*2, 1)).T
    '''
    print(f'use {time.time() - tmp} s')

    print(f'the basis shape is {X.shape}, and the dependent variable shape is {y.shape}')
    
    import statsmodels.api as sm
    if mode == 'PE':
        model = sm.GLM(y, X, family=sm.families.Poisson(), fit_intercept=False)
        result = model.fit()
        AIC = result.aic
        coef_ = result.params
        std = result.bse
    elif mode == 'time':
        import pandas as pd
        data = pd.DataFrame(data = np.hstack((X, np.atleast_2d(y).T)))
        strs = 'y ~ '
        start = data.keys().start
        stop = data.keys().stop
        step = data.keys().step

        cname = []
        cname.append('X0')
        for i in np.arange(start+1, stop, step):
            if i == start + 1:
                strs += 'X%d ' % i
            elif i == stop - step:
                pass
            else:
                strs += ' + X%d ' % i
            if i == stop - step:
                cname.append('y')
            else:
                cname.append('X%d' % i)
        data.columns = cname

        mod = sm.formula.quantreg(strs, data[cname])

        result = mod.fit(q=qt,)
        coef_ = result.params
        AIC = np.zeros_like(coef_)
        std = np.zeros_like(coef_)           
        print('Waring! No AIC and std value')
    print(result.summary())
    

    L, K = 500, 500
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    vertex = np.vstack((xv.flatten(), yv.flatten(), np.zeros_like(xv).flatten())).T
    PMTPos1 = np.tile(np.array((1,0,0)), (vertex.shape[0],1))
    X_fig, cos_theta = pub.LegendreCoeff(PMTPos1, vertex, order1, Legendre=True)
    rho = np.linalg.norm(vertex, axis=1)
    X1_fig = pub.legval(rho, np.eye(2*order2).reshape(2*order2,2*order2,1)).T
    LL = np.zeros((X1_fig.shape[0], order1*order2))
    for x1 in np.arange(order1):
        if not x1 % 5:
            print(x1)
        for x2 in np.arange(0, 2*order2,2):
            LL[:, x1*order2+np.int(x2/2)] = X_fig[:, x1]*X1_fig[:, x2]
            print(x1*order2+np.int(x2/2))
    X = LL
    data = np.dot(X, np.atleast_2d(coef_).T)
    data[rho>0.99] = np.nan
    plt.figure()
    data = np.clip(data, -2, 2)
    im = plt.imshow(data.reshape(K,L), origin='lower', extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.savefig('new_double_log_%d_%d.png' % (order, order2))
    plt.close()
    plt.figure()
    im = plt.imshow(np.exp(data).reshape(K,L), origin='lower', extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.savefig('new_double_%d_%d.png' % (order, order2))
    plt.close()

    with h5py.File(output, 'w') as opt:
        table = opt.create_dataset(f"coeff", data=coef_)
        table.attrs["t_min"] = -20
        table.attrs["t_max"] = 500
        table.attrs["type"] = "marginal"
        table.attrs["std"] = std
        table.attrs["AIC"] = AIC

parser = argparse.ArgumentParser(description='Process template construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.h5]', type=str,
                    help='The filename [*Q.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('--mode', dest='mode', type=str, choices=['PE', 'time'], default='PE',
                    help='Which info should be used')

parser.add_argument('--order', dest='order', metavar='N', type=int, default=10,
                    help=textwrap.dedent('''The max cutoff order. 
                    For Zernike is (N+1)*(N+2)/2'''))

parser.add_argument('--offset', dest='offset', metavar='filename[*.h5]', type=str, default=False,
                    help='Whether use offset data, default is 0')

parser.add_argument('--qt', type=float, default=0.1, 
                    help='quantile value')

args = parser.parse_args()
print(args.filename)

main_Calib(args.filename, args.output, args.mode, args.order, args.offset, args.qt)
