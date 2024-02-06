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

result_mcmc = pd.DataFrame(columns=['EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'r', 'xy'])
result_bc = pd.DataFrame(columns=['EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'r', 'xy'])

for f in args.ipt:
    with h5py.File(f,"r") as ipt:
        recon = pd.DataFrame(ipt['Recon'][:])
        reconbc = pd.DataFrame(ipt['ReconBC'][:])
        recon['r'] = np.sqrt(recon['x']**2 + recon['y']**2 + recon['z']**2)
        recon['xy'] = recon['x']**2 + recon['y']**2
        reconbc['r'] = np.sqrt(reconbc['x']**2 + reconbc['y']**2 + reconbc['z']**2)
        reconbc['xy'] = reconbc['x']**2 + reconbc['y']**2
        recon_data = recon.groupby('EventID').mean().reset_index()
        reconbc_data = reconbc.groupby('EventID').mean().reset_index()
        result_mcmc = pd.concat([result_mcmc, recon_data])
        result_bc = pd.concat([result_bc, reconbc_data])

mcmc = pd.merge(result_mcmc, eids[['EventID', 'particle']], on='EventID', how='inner')
bc = pd.merge(result_bc, eids[['EventID', 'particle']], on='EventID', how='inner')
mcmc['EventID'] = mcmc['EventID'].astype('int')
bc['EventID'] = bc['EventID'].astype('int')

# print 马尔科夫链接收率

opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("mcmc", data=mcmc.to_records(index=False), **opts)
    opt.create_dataset("bc", data=bc.to_records(index=False), **opts)
