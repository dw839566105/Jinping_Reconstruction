#!/usr/bin/env python3
'''
Prepare data for figure
'''
import pandas as pd
import argparse
import h5py
import numpy as np
import uproot
from DetectorConfig import shell

psr = argparse.ArgumentParser()
psr.add_argument("-e", dest="events", help="list of events")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('-r', dest='recon', nargs="+", help="recon file")
psr.add_argument('-b', dest='bc', help="bc file")
psr.add_argument("-s", dest="step", type=int, help="MC step")
args = psr.parse_args()

eids = pd.read_csv(args.events, sep='\s+', names=("particle", "run", "EventID", "file"),)
# 将'alpha'替换为0，'beta'替换为1
eids['particle'] = eids['particle'].replace({'alpha': 0, 'beta': 1})
eids = eids[['particle', 'EventID']].drop_duplicates()

def read_recon(inputfiles):
    print("read recon")
    result_mcmc = pd.DataFrame(columns=['EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'acceptr', 'acceptt', 'acceptz', 'r', 'xy'])
    for f in range(len(inputfiles)):
        with h5py.File(inputfiles[f],"r") as ipt:
            recon = pd.DataFrame(ipt['Recon'][:])
            # burn 前 3/5
            recon = recon[recon['step'] > (args.step / 5 * 3)]
            recon_data = recon.groupby('EventID').mean().reset_index()
            recon_data['x'] = recon_data['x'].apply(lambda x: x * shell)
            recon_data['y'] = recon_data['y'].apply(lambda x: x * shell)
            recon_data['z'] = recon_data['z'].apply(lambda x: x * shell)
            recon_data['r'] = np.sqrt(recon_data['x'].values**2 + recon_data['y'].values**2 + recon_data['z'].values**2)
            recon_data['xy'] = np.sqrt(recon_data['x'].values**2 + recon_data['y'].values**2)
            result_mcmc = pd.concat([result_mcmc, recon_data])
    result_mcmc['EventID'] = result_mcmc['EventID'].astype('int')
    print("done")
    return result_mcmc

def read_bc(inputfile):
    print("read bc")
    PEs = 65
    prompt_s = 0.5
    prompt_e = 3.5
    delay_s = 0.4
    delay_e = 1.2
    tcut = 0.0004
    dcut = 400
    f = uproot.open(inputfile)
    Fold = f['Event']['Fold'].array()
    # 时间窗内有两个事例
    cut1 = (Fold == 2)
    E = f['Event']['E'].array()[cut1]
    X = f['Event']['X'].array()[cut1]
    Y = f['Event']['Y'].array()[cut1]
    Z = f['Event']['Z'].array()[cut1]
    TrigSec = f['Event']['TrigSec'].array()[cut1]
    TrigNano = f['Event']['TrigNano'].array()[cut1]
    TrigNum = f['Event']['TrigNum'].array()[cut1]
    FileNum = f['Event']['FileNum'].array()[cut1]
    Time = f['Event']['T2PrevSubEvt'].array()[cut1]
    D2First = f['Event']['D2First'].array()[cut1]
    f.close()
    cut2 = (E[:,0] > prompt_s * PEs) * (E[:,0] < prompt_e * PEs) * (E[:,1] > delay_s * PEs) * (E[:,1] < delay_e * PEs)
    cut3 = (Time[:,1] < tcut)
    cut4 = (D2First[:,1] < dcut)
    cut = cut2 * cut3

    result_alpha = pd.DataFrame(columns=['EventID', 'x', 'y', 'z', 'E', 'r', 'xy'])
    result_alpha['EventID'] = TrigNum[cut][:,1]
    result_alpha['x'] = X[cut][:,1] / 1000
    result_alpha['y'] = Y[cut][:,1] / 1000
    result_alpha['z'] = Z[cut][:,1] / 1000
    result_alpha['E'] = E[cut][:,1]
    result_alpha['r'] = np.sqrt(result_alpha['x']**2 + result_alpha['y']**2 + result_alpha['z']**2)
    result_alpha['xy'] = np.sqrt(result_alpha['x']**2 + result_alpha['y']**2)
    result_beta = pd.DataFrame(columns=['EventID', 'x', 'y', 'z', 'E', 'r', 'xy'])
    result_beta['EventID'] = TrigNum[cut][:,0]
    result_beta['x'] = X[cut][:,0] / 1000
    result_beta['y'] = Y[cut][:,0] / 1000
    result_beta['z'] = Z[cut][:,0] / 1000
    result_beta['E'] = E[cut][:,0]
    result_beta['r'] = np.sqrt(result_beta['x']**2 + result_beta['y']**2 + result_beta['z']**2)
    result_beta['xy'] = np.sqrt(result_beta['x']**2 + result_beta['y']**2)
    result_bc = pd.concat([result_alpha, result_beta])
    print("done")
    return result_bc   

bc = pd.merge(read_bc(args.bc), eids[['EventID', 'particle']], on='EventID', how='inner')
recon = pd.merge(read_recon(args.recon), eids[['EventID', 'particle']], on='EventID', how='inner')
opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("bc", data=bc.to_records(index=False), **opts)
    opt.create_dataset("recon", data=recon.to_records(index=False), **opts)

