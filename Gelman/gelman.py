#!/usr/bin/env python3
'''
merge different chains into one dataframe
'''
import h5py
import numpy as np
import argparse
import pandas as pd

psr = argparse.ArgumentParser()
psr.add_argument("-i", dest="ipt", type=str, nargs='+', help="input files")
psr.add_argument("-o", dest="opt", type=str, help="output file")
args = psr.parse_args()

data = []
for inputf in args.ipt:
    with h5py.File(inputf, "r") as f:
        recon = pd.DataFrame(f["Recon"][:])
        recon = recon.drop('Likelihood', axis=1)
        acc = recon.groupby('EventID').sum().reset_index()
        print(acc)
        recon = recon.drop('accept', axis=1)
        data.append(recon)

merged_df = data[0]
for i in range(1, len(data)):
    merged_df = pd.merge(merged_df, data[i], on=['EventID', 'step'], suffixes=('', '_{}'.format(i+1)))

opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("data", data=merged_df.to_records(index=False), **opts)