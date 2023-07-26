#!/usr/bin/python
# -*- coding:utf-8 -*-
import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import argparse, textwrap
from argparse import RawTextHelpFormatter
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(description='Recover 1-d Legendre', formatter_class=RawTextHelpFormatter)
parser.add_argument('-p', '--path', dest='path', metavar='path', type=str,
                    help='The filename [path] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('--o1', dest='order1', metavar='order1', type=int,
                    help='The order to be used')

parser.add_argument('--o2', dest='order2', metavar='order2', type=int,
                    help='The order to be fitted')
args = parser.parse_args()

def LoadDataPE_TW(filename):
    with tables.open_file(filename,'r') as h:
        d1 = h.root.coeff[:][:, np.newaxis]
        d2 = h.root.coeff.attrs['std'][:, np.newaxis]
    return d1, d2

def gather_coeff(path, ra, order, mode):
    coeff = []
    std = []
    for r_index, radius in enumerate(ra):
        if mode == 'compact':
            str_radius = '%.3f' % radius
        elif mode == 'sparse':
            str_radius = '%.2f' % radius
        filename = '{}/{}/{:02d}.h5'.format(path, str_radius, order)
        d1, d2 = LoadDataPE_TW(filename)
        if(r_index == 0):
            coeff, std = d1, d2
        else:
            coeff = np.hstack((coeff, d1))
            std = np.hstack((std, d2))
    return coeff, std

def main(path, order=5, fit_order=10):
    ra1 = np.arange(0.01,0.56,0.01)
    coeff1, std1 = gather_coeff(path, ra1, order, 'sparse')
    ra2 = np.arange(0.55,0.64,0.002)
    coeff2, std2 = gather_coeff(path, ra2, order, 'compact')
    
    rd = np.hstack((ra1, ra2))
    coeff = np.hstack((coeff1, coeff2))
    coeff_fit = np.zeros((order, fit_order + 1))
    
    with PdfPages(args.output + '.pdf') as pdf:
        for i in range(order):
            if not i%2:
                B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))), \
                                                          np.hstack((coeff[i], coeff[i])), \
                                                          deg = fit_order, full = True)
            else:
                B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))), \
                                                          np.hstack((coeff[i], -coeff[i])), \
                                                          deg = fit_order, full = True)

            y = np.polynomial.legendre.legval(rd/np.max(rd), B)

            coeff_fit[i] = B

            fig = plt.figure(dpi = 300)
            ax = plt.gca()
            ax.plot(rd, coeff[i], 'r.', label='real',linewidth=2)
            ax.plot(rd, y, label = 'Legendre')
            ax.set_xlabel('radius/m')
            ax.set_ylabel('PE Legendre coefficients')
            ax.set_title('%d th, max fit = %d' %(i, fit_order))
            ax.legend()
            pdf.savefig(fig)
            plt.close()
    return coeff_fit

coeff_fit = main(args.path, args.order1, args.order2)  
with h5py.File(args.output,'w') as out:
    table = out.create_dataset('coeff', data = coeff_fit)
    table.attrs["t_min"] = -20 
    table.attrs["t_max"] = 500 
    table.attrs["type"] = "Legendre"
    table.attrs["order"] = args.order1
