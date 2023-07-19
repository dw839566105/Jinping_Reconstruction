import numpy as np
import tables
import pandas as pd
from polynomial import *
from zernike import RZern
from numpy.polynomial import legendre as LG
from argparse import RawTextHelpFormatter
import argparse, textwrap

r_max = 638
parser = argparse.ArgumentParser(description='Process template construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('--pe', dest='pe', metavar='PE[*.h5]', type=str,
                    help='The pe coefficient [.h5] to read')

parser.add_argument('--time', dest='time', metavar='Time[*.h5]', type=str,
                    help='The time coefficient [*.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='Coeff[*.h5]', type=str,
                    help='The output file [*.h5] to save')

args = parser.parse_args()

def loadh5(filename):
    h = tables.open_file(filename)
    coef_ = h.root.coeff[:]
    # coef_type = 'Zernike'
    coef_type = h.root.coeff.attrs.type
    h.close()
    return coef_, coef_type

coef_PE, PE_type = loadh5(args.pe)
coef_Time, Time_type = loadh5(args.time)

def calc_probe(r, theta, coef, coef_type):
    if(coef_type=='Zernike'):
        cart = RZern(20)
        zo = cart.mtab>=0
        zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], r.flatten())
        zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), theta.flatten())
        probe = np.matmul((zs_radial * zs_angulars)[:len(coef)].T, coef)
    elif(coef_type=='Legendre'):
        cut = len(coef)
        t_basis = legval_raw(np.cos(theta), np.eye(cut).reshape((cut,cut,1))).T
        r_basis = legval_raw(r, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
        probe = (r_basis*t_basis).sum(-1)
    elif(coef_type=='db_Legendre'):
        o1, o2 = coef.shape
        X1 = legval_raw(r, np.eye(2*o1).reshape((2*o1, 2*o1, 1))).T
        X2 = legval_raw(np.cos(theta), np.eye(o2).reshape((o2, o2, 1))).T
        X = np.empty((len(X1), o1*o2))
        for i in range(o1):
            for j in range(o2):
                X[:,i*o2 + j] = X1[:,i] * X2[:,j]
        probe = np.dot(X, coef.flatten())
    return probe.reshape(r.shape)

def concat():
    for index, i in enumerate(np.arange(50,60)):
        h = tables.open_file(f'/mnt/stage/douwei/JP_1t_paper/concat/ball/2/{i:02d}.h5')
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

def quantile(t, tau, ts, expect):
    return tau*(1-tau)/ts * np.exp(-1/ts *((1 - tau)* (expect - t) * (expect > t) + tau* (t- expect) * (t >= expect)))

df_hit, df_nonhit = concat()
expect_PE_r = calc_probe(np.clip(df_nonhit['r'].values/r_max, 0, 1), df_nonhit['theta'].values, coef_PE, PE_type)
expect_PE_h = calc_probe(np.clip(df_hit['r'].values/r_max, 0, 1), df_hit['theta'].values, coef_PE, PE_type)

if (PE_type=='Zernike'):
    score_PE = np.sum(-np.exp(expect_PE_r[df_nonhit['r'].values/r_max<1])) + np.sum(expect_PE_h[df_hit['r'].values/r_max<1])
elif (PE_type=='db_Legendre'):
    score_PE = np.sum(-np.exp(expect_PE_r[df_nonhit['r'].values/r_max<1])) + np.sum(expect_PE_h[df_hit['r'].values/r_max<1])
elif (PE_type=='Legendre'):
    score_PE = np.sum(-np.exp(expect_PE_r)) + np.sum(expect_PE_h)

print(score_PE)
df = pd.DataFrame(np.array([score_PE,])[:, np.newaxis].T, columns=['score',])
df.T.to_csv(args.output, header=False)

# time part if needed in future
'''
expect_time_h = calc_probe(np.clip(df_hit['r'].values/r_max, 0, 1), df_hit['theta'].values, coef_Time, Time_type)
LH_time = quantile(df_hit['t'], tau=0.1, ts=3, expect = expect_time_h)
if (Time_type=='Zernike'):
    score_Time = np.sum(np.log(LH_time[df_hit['r'].values/r_max<1]))
elif (Time_type=='db_Legendre'):
    score_Time = np.sum(np.log(LH_time[df_hit['r'].values/r_max<1]))
elif (Time_type=='Legendre'):
    score_Time = np.sum(np.log(LH_time))
score = score_PE # + score_Time
print(score)
df = pd.DataFrame(np.array([score_PE, score_Time, score])[:, np.newaxis].T, columns=['PE', 'time', 'score'])
df.T.to_csv(args.output, header=False)
'''