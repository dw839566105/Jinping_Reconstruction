import pub
import numpy as np

import h5py
import argparse
from argparse import RawTextHelpFormatter
import argparse, textwrap
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
from zernike import RZern
import pandas as pd


def main_Calib(filename, output, mode, order, offset, qt):

    print('begin reading file', flush=True)
    if(offset):
        off = pub.LoadBase(offset)
    else:
        off = np.zeros_like(PMTPos[:,0])

    tmp = time.time()

    ## sphere
    for i in np.arange(1,11):
        if i == 1:
            with h5py.File(filename + '%02d.h5' % i, "r") as ipt:
                h_hit = ipt['Concat'][()]
                h_nonhit = ipt['Vertices'][()]
        else:
            with h5py.File(filename + '%02d.h5' % i, "r") as ipt:
                data = ipt['Concat'][()]
                data['EId'] += np.max(h_hit['EId'])
                h_hit = np.hstack((h_hit, data))
                h_nonhit = np.hstack((h_nonhit, ipt['Vertices'][()]))

    EventId = h_hit['EId']
    ChannelId = h_hit['CId']

    if mode == 'PE':
        y, _, _ = np.histogram2d(EventId, ChannelId,
            bins = (np.arange(np.min(EventId), np.max(EventId)+2)-0.5, np.arange(len(PMTPos)+1) - 0.5))
        y = y.flatten()
        rho = h_nonhit['r']/1000/args.r_max
        theta = h_nonhit['theta']
    elif mode == 'time':
        y = h_hit['t']
        rho = h_hit['r']/1000/args.r_max
        theta = h_hit['theta']

    cart = RZern(order)
    nk = cart.nk
    m = cart.mtab
    n = cart.ntab
    X = np.zeros((rho.shape[0], nk))
    for i in np.arange(nk):
        if not i % 5:
            print(f'process {i}-th event')
        X[:,i] = cart.Zk(i, rho, theta)
    X = X[:,m>=0]

    print(f'use {time.time() - tmp} s')
    print(f'the basis shape is {X.shape}, and the dependent variable shape is {y.shape}')

    if mode == 'PE':
        model = sm.GLM(y, X, family=sm.families.Poisson(), fit_intercept=False)
        result = model.fit()
        AIC = result.aic
        coef_ = result.params
        std = result.bse
    elif mode == 'time':
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

    with h5py.File(output, 'w') as opt:
        table = opt.create_dataset(f"coeff", data=coef_)
        table.attrs["t_min"] = -20
        table.attrs["t_max"] = 500
        table.attrs["type"] = "Zernike"
        table.attrs["order"] = order
        table.attrs["std"] = std
        table.attrs["AIC"] = AIC

parser = argparse.ArgumentParser(description='Process template construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.h5]', type=str,
                    help='The filename [*Q.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('--mode', dest='mode', type=str, choices=['PE', 'time'], default='PE',
                    help='Which info should be used')

parser.add_argument('--pmt', dest='pmt', type=str, default='./PMT.txt',
                    help='Which info should be used')

parser.add_argument('--order', dest='order', metavar='N', type=int, default=10,
                    help=textwrap.dedent('''The max cutoff order. 
                    For Zernike is (N+1)*(N+2)/2'''))

parser.add_argument('--offset', dest='offset', metavar='filename[*.h5]', type=str, default=False)

parser.add_argument('--r_max', type=float, default=1.3,
                    help='maximum LS radius')

parser.add_argument('--qt', type=float, default=0.1, 
                    help='quantile value')

args = parser.parse_args()
print(args.filename)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# global:
PMTPos = pub.ReadJPPMT(args.pmt)
main_Calib(args.filename, args.output, args.mode, args.order, args.offset, args.qt)
    
