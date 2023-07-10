import tables
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial import legendre as LG
import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description='Process template construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-p', '--path', dest='path', metavar='path', type=str,
                    help='The path to read')
parser.add_argument('-m', '--mode', dest='mode', metavar='mode', type=str,
                    help='The path to read')
parser.add_argument('-o', '--order', dest='order', metavar='N[int]', type=int,
                    help='order')
args = parser.parse_args()

def readfile(strs):
    for index, i in enumerate(rd):
        h = tables.open_file(f'{strs}/{i:.2f}.h5')
        if index == 0:
            data = pd.DataFrame(h.root.coeff[:][:,np.newaxis].T)
        else:
            data = pd.concat([data, pd.DataFrame(h.root.coeff[:][:,np.newaxis].T)])
        h.close()
    return data

rd = np.arange(0.01,0.64,0.01)
deg = 40
strs = f'{args.path}/{args.mode}/{args.order:02d}'
data = readfile(strs)

coef_mat = np.empty((len(data.keys()), deg + 1))
for index, i in enumerate(np.arange(len(data.keys()))):
    x = rd/0.65
    y = data[pd.RangeIndex(start=i, stop = i+1)].values

    X = np.hstack((x, -x))
    if not i%2:
        Y = np.vstack((y, y))
    else:
        Y = np.vstack((y, -y))
    B = LG.legfit(X, Y, deg = deg)
    res = LG.legval(x, B)
    coef_mat[index] = B.flatten()

with h5py.File(f'{strs}/total.h5','w') as out:
    out['coeff'] = coef_mat
    out['coeff'].attrs["type"] = "Legendre"
