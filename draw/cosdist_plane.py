import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import jit

import matplotlib.pyplot as plt
import tables
import numpy as np

from tqdm import tqdm

import sys
radius = eval(sys.argv[1])

@jit(nopython=True)
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
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
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x

h = tables.open_file('/home/douwei/ReconJP/coeff/Legendre/Gather/PE/2/70/50.h5','r')
coeff = h.root.coeff[:]
h.close()

cut, fitcut = coeff.shape
PMT_pos = np.loadtxt('/home/douwei/ReconJP/PMT_1t.txt')

## tpl

N = 300
xr = np.linspace(-1, 1, N)
yr = np.linspace(-1, 1, N)

        
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('scan_2d_new1.pdf') as pp:
    vertex = np.array((radius/0.638*np.cos(0), 0, 0)) # x axis
    cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)
    c = np.eye(cut).reshape((cut, cut, 1))
    x = legval(cos_theta, c).T
    k = legval(np.linalg.norm(vertex), coeff.T)
    expect0 = np.exp(np.dot(x, k))

    plate = np.empty((N, N, 30))
    for x_index, x in enumerate(tqdm(xr)):
        for y_index, y in enumerate(yr):
            vertex = np.array((x, y, 0))
            cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)

            xx = legval(cos_theta, c).T
            k = legval(np.sqrt(x**2+y**2), coeff.T)

            expect = np.exp(np.dot(xx,k))
            if(x**2+y**2>1):
                expect[:] = np.nan
            plate[x_index, y_index] = expect

    dist = np.zeros((len(plate), len(plate[0])))
    for i in tqdm(np.arange(len(plate))):
        for j in np.arange(len(plate[0])):
            d = np.sum(expect0*plate[i][j])/np.linalg.norm(expect0)/np.linalg.norm(plate[i][j])
            dist[i,j] = d

    plt.figure(dpi=300, figsize=(6,4.5))
    x_new, y_new = np.meshgrid(xr/0.65,yr/0.65, sparse=False)
    im = plt.contourf(xr*0.65*1000, yr*0.65*1000, 1-dist.T, origin='lower', levels=50, cmap='gray')
    plt.scatter(radius*1000,0,c='r', label='truth', marker='*', s=50)
    plt.scatter(PMT_pos[:,0]*1000, PMT_pos[:,1]*1000, c='k', label='PMT', marker='^', s=50)
    plt.legend(fontsize=20, facecolor='w', loc='lower right')
    plt.ylabel('$y$/mm')
    plt.xlabel('$x$/mm')
    plt.gca().tick_params(labelsize=18)
    plt.gca().set_aspect(1)
    plt.ylim([-0.83*1000, 0.83*1000])
    plt.xlim([-0.83*1000, 0.83*1000])
    cbar = plt.colorbar(im, cmap='jet',ticks=np.arange(0,0.8,0.1), format='%.2f', pad=0.01)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)
    pp.savefig()

    plt.figure(dpi=300, figsize=(6,4.5))
    x_new, y_new = np.meshgrid(xr/0.65,yr/0.65, sparse=False)
    im = plt.contourf(xr*0.65*1000, yr*0.65*1000, 1-dist.T, origin='lower', levels=np.arange(0,0.21,0.01), cmap='gray')
    plt.scatter(radius*1000,0,c='r', label='truth', marker='*', s=50)
    plt.legend(fontsize=20, facecolor='w', loc='lower right')
    plt.gca().tick_params(labelsize=18)
    plt.ylabel('$y$/mm')
    plt.xlabel('$x$/mm')
    plt.plot(0.57*1000, 0.13*1000, 'o', ms=30, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=1)
    plt.plot(0.57*1000, -0.13*1000, 'o', ms=30, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=1)
    plt.gca().set_aspect(1)
    plt.ylim([-0.2*1000, 0.2*1000])
    plt.xlim([0.2*1000, 0.6*1000])
    cbar = plt.colorbar(im, cmap='jet',ticks=np.arange(0,0.8,0.1), format='%.2f', pad=0.01)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)
    pp.savefig()
    
    vertex = np.array((0, radius/0.638, 0)) # y axis
    cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)
    c = np.eye(cut).reshape((cut, cut, 1))
    x = legval(cos_theta, c).T
    k = legval(np.linalg.norm(vertex), coeff.T)
    expect0 = np.exp(np.dot(x, k))
    
    plate = np.empty((N, N, 30))
    for x_index, x in enumerate(tqdm(xr)):
        for y_index, y in enumerate(yr):
            vertex = np.array((0, x, y))
            cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)

            xx = legval(cos_theta, c).T
            k = legval(np.sqrt(x**2+y**2), coeff.T)

            expect = np.exp(np.dot(xx,k))
            if(x**2+y**2>1):
                expect[:] = np.nan
            plate[x_index, y_index] = expect

    dist = np.zeros((len(plate), len(plate[0])))
    for i in tqdm(np.arange(len(plate))):
        for j in np.arange(len(plate[0])):
            d = np.sum(expect0*plate[i][j])/np.linalg.norm(expect0)/np.linalg.norm(plate[i][j])
            dist[i,j] = d

    plt.figure(dpi=300, figsize=(6,4.5))
    x_new, y_new = np.meshgrid(xr/0.65, yr/0.65, sparse=False)
    im = plt.contourf(xr*0.65*1000, yr*0.65*1000, 1-dist.T, origin='lower', levels=50, cmap='gray')
    plt.scatter(radius*1000,0,c='r', label='truth', marker='*', s=50)
    plt.scatter(PMT_pos[:,1]*1000, PMT_pos[:,2]*1000, c='k', label='PMT', marker='^', s=50)
    plt.legend(fontsize=20, facecolor='w', loc='lower right')
    plt.gca().tick_params(labelsize=18)
    plt.xlabel('$y$/mm')
    plt.ylabel('$z$/mm')
    plt.gca().set_aspect(1)
    plt.ylim([-0.83*1000, 0.83*1000])
    plt.xlim([-0.83*1000, 0.83*1000])
    cbar = plt.colorbar(im, cmap='jet',ticks=np.arange(0,0.8,0.1), format='%.2f', pad=0.01)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)
    pp.savefig()

    plt.figure(dpi=300, figsize=(6,4.5))
    x_new, y_new = np.meshgrid(xr/0.65, yr/0.65, sparse=False)
    im = plt.contourf(xr*0.65*1000, yr*0.65*1000, 1-dist.T, origin='lower', levels=np.arange(0,0.21,0.01), cmap='gray')
    plt.scatter(radius*1000,0,c='r', label='truth', marker='*', s=50)
    plt.legend(fontsize=20, facecolor='w', loc='lower right')
    plt.gca().tick_params(labelsize=18)
    plt.xlabel('$y$/mm')
    plt.ylabel('$z$/mm')
    plt.plot(0.57*1000, 0, 'o', ms=30, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=1)
    plt.gca().set_aspect(1)
    plt.ylim([-0.2*1000, 0.2*1000])
    plt.xlim([0.2*1000, 0.6*1000])
    cbar = plt.colorbar(im, cmap='jet',ticks=np.arange(0,0.8,0.1), format='%.2f', pad=0.01)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)
    pp.savefig()