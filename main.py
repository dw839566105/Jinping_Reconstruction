#!/usr/bin/env python3
'''
重建模拟数据的位置、能量
'''
import argparse
from argparse import RawTextHelpFormatter
import Recon
import Detector
import numpy as np
import pandas as pd
    
parser = argparse.ArgumentParser(description='Process Reconstruction construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.pq]', type=str,
                    help='The filename [*Q.pq] to read')

parser.add_argument('--sparsify', dest='sparsify', metavar='sparsify[*.h5]', type=str, 
                    default=None, help='The filename [*Q.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('-p', '--probe', dest='probe', metavar='probe[*.h5]', type=str,
                    default=None, help='The probe filename [*.h5] to read')

parser.add_argument('--pe', dest='pe', metavar='PECoeff[*.h5]', type=str, 
                    default=None, help='The pe coefficients file [*.h5] to be loaded')

parser.add_argument('--time', dest='time', metavar='TimeCoeff[*.h5]', type=str,
                    default=None, help='The time coefficients file [*.h5] to be loaded')

parser.add_argument('--PMT', dest='PMT', metavar='PMT[*.txt]', type=str, 
                    help='The PMT file [*.txt] to be loaded')

parser.add_argument('--dark', dest='dark', type=str,
                    help='dark rate file')

parser.add_argument('--timecalib', dest='timecalib', type=str,
                    help='time calib file')

parser.add_argument('-n', '--num', dest='num', type=int, default=10000,
                    help='test event nums')

parser.add_argument('--record', dest='record', type=str, default="OFF",
                    help='record each step sampling ON/OFF')

parser.add_argument('-m', '--MCstep', dest='MCstep', type=int, default=10,
                    help='mcmc step per PEt')

args = parser.parse_args()

# 读入文件
pmt_pos = np.loadtxt(args.PMT)
print("Finished Reading PMT")
probe = Detector.LoadProbe(args.probe, args.pe, args.time, pmt_pos)
print("Finished Loading Probe")
timefile = pd.read_csv(args.timecalib, sep='\s+', header=None, comment="#")
darkrate = np.loadtxt(args.dark) / 1E9 # Hz 转换成 个 / ns
timecalib = - timefile[6].values

# 重建
Recon.reconstruction(args.filename, args.sparsify, args.num, args.output, probe, pmt_pos, darkrate, timecalib, args.MCstep, args.record)


