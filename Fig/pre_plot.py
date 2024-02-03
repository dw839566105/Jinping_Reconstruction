#!/usr/bin/env python3
'''
Prepare data for figure
'''
import pandas as pd
import argparse
import h5py
import numpy as np

psr = argparse.ArgumentParser()
psr.add_argument("-e", dest="events", help="list of events")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', nargs="+", help="input")
args = psr.parse_args()

eids = pd.read_csv(args.events, sep='\s+', names=("particle", "run", "EventID", "file"),)
# 将'alpha'替换为0，'beta'替换为1
eids['particle'] = eids['particle'].replace({'alpha': 0, 'beta': 1})
eids = eids[['particle', 'EventID']].drop_duplicates()

def Getdata(data):
    data['r'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['xy'] = np.sqrt(data['x']**2 + data['y']**2)
    average_r = data.groupby('EventID')['r'].mean().reset_index()
    average_E = data.groupby('EventID')['E'].mean().reset_index()
    average_z = data.groupby('EventID')['z'].mean().reset_index()
    average_xy = data.groupby('EventID')['xy'].mean().reset_index()
    max_freq_E = data.groupby('EventID').apply(lambda x: x['E'].value_counts(bins=100).idxmax().mid).reset_index()
    max_freq_E.columns = ['EventID', 'E_max']
    average_E.columns = ['EventID', 'E_mean']
    average_r.columns = ['EventID', 'r_mean']
    average_z.columns = ['EventID', 'z_mean']
    average_xy.columns = ['EventID', 'xy_mean']
    result = pd.merge(pd.merge(pd.merge(pd.merge(average_r, average_E, on='EventID'), max_freq_E, on='EventID'), average_z, on='EventID'), average_xy, on='EventID')
    merged_df = pd.merge(result, eids[['EventID', 'particle']], on='EventID', how='inner')
    return merged_df

result_mcmc = pd.DataFrame(columns=['EventID', 'E_max', 'E_mean', 'r_mean', 'z_mean', 'xy_mean'])
result_bc = pd.DataFrame(columns=['EventID', 'E_max', 'E_mean', 'r_mean', 'z_mean', 'xy_mean'])

for f in args.ipt:
    with h5py.File(f,"r") as ipt:
        recon = pd.DataFrame(ipt['Recon'][:])
        reconbc = pd.DataFrame(ipt['ReconBC'][:])
        recon_data = Getdata(recon)
        reconbc_data = Getdata(reconbc)
        result_mcmc = pd.concat([result_mcmc, recon_data])
        result_bc = pd.concat([result_bc, reconbc_data])

result_mcmc['EventID'] = result_mcmc['EventID'].astype('int')
result_bc['EventID'] = result_bc['EventID'].astype('int')
opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("mcmc", data=result_mcmc.to_records(index=False), **opts)
    opt.create_dataset("bc", data=result_bc.to_records(index=False), **opts)
