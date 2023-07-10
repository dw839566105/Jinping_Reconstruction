import pub
import numpy as np
from polynomial import *
import h5py
import argparse
from argparse import RawTextHelpFormatter
import argparse, textwrap
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
from zernike import RZern
import pandas as pd

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

    ## shell
    for idx, i in enumerate(np.arange(0.01, 0.55, 0.01)):
        print(i)
        if idx == 0:
            with h5py.File(filename + '%.2f.h5' % i, "r") as ipt:
                h_hit = ipt['Concat'][()]
                h_nonhit = ipt['Vertices'][()]
        else:
            with h5py.File(filename + '%.2f.h5' % i, "r") as ipt:
                data = ipt['Concat'][()]
                data['EId'] += np.max(h_hit['EId'])
                h_hit = np.hstack((h_hit, data))
                h_nonhit = np.hstack((h_nonhit, ipt['Vertices'][()]))

    for i in np.arange(0.55, 0.640, 0.002):
        with h5py.File(filename + '%.3f.h5' % i, "r") as ipt:
            print(i)
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
        
    with h5py.File('/mnt/stage/douwei/JP_1t_paper_check/coeff/Legendre/Gather/PE/2/40/30.h5') as h:
        coef1 = h['coeff'][:][:,:-1]
    
    o1, o2 = coef1.shape
    #X1 = pub.legval(rho, np.eye(2*o1).reshape((2*o1, 2*o1, 1))).T
    #X1 = X1[:,::2]
    #X2 = pub.legval(np.cos(theta), np.eye(o2).reshape((o2, o2, 1))).T

    #X = np.empty((len(X1), o1*o2))
    #for i in range(o1):
    #    for j in range(o2):
    #        X[:,i*o2 + j] = X1[:,i] * X2[:,j]
    X1 = legval_raw(rho, np.eye(o2).reshape((o2, o2, 1))).T
    X2 = legval_raw(np.cos(theta), np.eye(o1).reshape((o1, o1, 1))).T
    X = np.empty((len(X1), o2*o1))
    for i in range(o2):
        for j in range(o1):
            X[:,i*o1 + j] = X1[:,i] * X2[:,j]
            
    A = np.ones((o1, o2))
    A1 = A.copy()
    A2 = A.copy()
    A1[:, ::2] = 0
    A2[::2, :] = 0
    index = (A1 == A2).flatten()
    
    X = X[:,index]

    expect = X @ coef1.T.flatten()[index]
    np.exp(expect).sum() - (y*expect).sum()
    breakpoint()
    del X1, X2
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

    o1, o2 = coef1.shape
    coef_ = coef_.reshape(o1, o2)
    print(result.summary())

    with h5py.File(output, 'w') as opt:
        table = opt.create_dataset(f"coeff", data=coef_)
        table.attrs["t_min"] = -20
        table.attrs["t_max"] = 500
        table.attrs["type"] = "db_Legendre"
        table.attrs["std"] = std
        table.attrs["AIC"] = AIC

parser = argparse.ArgumentParser(description='Process template construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.h5]', type=str,
                    help='The filename [*Q.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('--mode', dest='mode', type=str, choices=['PE', 'time'], default='PE',
                    help='Which info should be used')

parser.add_argument('--order', dest='order', type=int, nargs='+',
                    )

parser.add_argument('--order2', dest='o2', metavar='N', type=int, default=10,
                    help=textwrap.dedent('''The max cutoff order. 
                    For Zernike is (N+1)*(N+2)/2'''))

parser.add_argument('--offset', dest='offset', metavar='filename[*.h5]', type=str, default=False)

parser.add_argument('--r_max', type=float, default=0.65,
                    help='maximum LS radius')

parser.add_argument('--qt', type=float, default=0.1, 
                    help='quantile value')

args = parser.parse_args()
print(args.filename)

main_Calib(args.filename, args.output, args.mode, args.order, args.offset, args.qt)
    
