import numpy as np
import tables
import matplotlib.pyplot as plt
import pandas as pd
from zernike import RZern
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from polynomial import *
from argparse import RawTextHelpFormatter
from tqdm import tqdm
import argparse

plt.rc('text', usetex=False)
parser = argparse.ArgumentParser(description='Process template construction', formatter_class=RawTextHelpFormatter)

parser.add_argument('--R', dest='R', metavar='R', type=float,
                            help='PMT radius(>1)')

parser.add_argument('--N', dest='N', metavar='N', type=int,
                            help='PMT Number')

args = parser.parse_args()

cart = RZern(30)
zo = cart.mtab>=0

def calc_probe(ddx, ddy):

    v = ddx * np.vstack((np.sin(ddy), np.cos(ddy))).T
    a0 = (v[:, 1] - PMT[:,1])/(v[:,0] - PMT[:, 0])
    a1 = - v[:, 0] * a0 + v[:, 1]

    a = a0**2 + 1
    b = 2*a0*a1
    c = a1**2 - 1**2

    delta = np.sqrt(b**2 - 4*a*c)
    x1 = (-b - delta)/2/a
    x2 = (-b + delta)/2/a
    intercept = np.vstack((x2, a0*x2 + a1)).T

    dist = np.linalg.norm(PMT - v, axis=1)
    cth = np.sum((intercept-v)*intercept, axis=1)/np.linalg.norm((intercept-v), axis=1)
    cth = np.nan_to_num(cth, nan=1)
    th1 = np.arccos(np.clip(cth, -1, 1))
    th2 = np.arcsin(np.sin(th1)*1.5/1.33)
    t_ratio = 2*1.47*cth/(1.47*cth + 1.33*np.cos(th2))
    tr = 1 - (t_ratio-1)**2
    probe = cth/dist**2*np.nan_to_num(tr)
    return probe

theta = np.linspace(-np.pi, np.pi, args.N + 1)[:-1]
PMT = args.R * np.vstack([np.sin(theta), np.cos(theta)]).T

thetas = np.linspace(0, 2 * np.pi, 1001)
rs = np.linspace(0.01, 0.99, 1001)

def local_minima(data):
    center = data[1:-1,1:-1]
    left = data[1:-1,:-2]
    right = data[1:-1,2:]
    up = data[:-2,1:-1]
    bottom = data[2:,1:-1]

    ul = data[:-2,:-2]
    ur = data[:-2,2:]
    bl = data[2:,:-2]
    br = data[2:,2:]
    idx1 = (center<left) & (center<right) & (center<up) & (center<bottom)
    idx2 = (center<ul) & (center<ur) & (center<bl) & (center<br)
    mins = idx1 & idx2
    return np.where(mins)


plate = np.empty((len(thetas), len(rs), len(PMT)))
for x_index, x in enumerate(tqdm(thetas)):
    for y_index, y in enumerate(rs):
        # vertex = np.array([y*np.sin(x), y*np.cos(x)])[:, np.newaxis].T # x axis
        # vertex = np.array((0, r*np.sin(t), r*np.cos(t))) # z axis
        # cth = np.sum(vertex*PMT, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT, axis=1)
        probe = calc_probe(y, x).T
        plate[x_index, y_index] = probe

X, Y = np.meshgrid(thetas, rs)

i = 900
data0 = plate[500][i]
dist = 1 - (data0*plate).sum(-1)/np.linalg.norm(data0)/np.linalg.norm(plate, axis=-1)
mins = local_minima(dist)
print(len(mins[0]))
print(mins)