#!/usr/bin/env python3
'''
Prepare data for figure
'''
import pandas as pd
import argparse
import h5py
import numpy as np
import uproot
import pyarrow.parquet as pq

shell = 0.65
psr = argparse.ArgumentParser()
psr.add_argument("-e", dest="events", help="list of events")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('--step', dest='step', nargs="+", help="step recon")
psr.add_argument('--stack', dest='stack', nargs="+", help="stack recon")
psr.add_argument('--fsmp', dest='fsmp', nargs="+", help="fsmp result")
psr.add_argument('--bc', dest='bc', help="bc recon")
psr.add_argument('--ser', dest='ser', help="ser file")
args = psr.parse_args()

eids = pd.read_csv(args.events, sep='\s+', names=("particle", "run", "EventID", "file"),)
# 将'alpha'替换为0，'beta'替换为1
eids['particle'] = eids['particle'].replace({'alpha': 0, 'beta': 1})
eids = eids[['particle', 'EventID']].drop_duplicates()

def read_recon(inputfiles, connect):
    print("read recon")
    result_mcmc = pd.DataFrame(columns=['EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'r'])
    for f in range(len(inputfiles)):
        with h5py.File(inputfiles[f],"r") as ipt:
            recon = pd.DataFrame(ipt['Recon'][:])
            recon['r'] = np.sqrt(recon['x']**2 + recon['y']**2 + recon['z']**2)
            if connect == "stack":
                recon['E'] = recon['E'].apply(lambda x: x / 2500)
            recon_data = recon.groupby('EventID').mean().reset_index()
            # std = recon.groupby('EventID').agg({'xy': 'std', 'z': 'std'})
            # std = std.rename(columns={'xy': 'std_xy', 'z': 'std_z'})
            # recon_data = pd.merge(recon_data, std, on='EventID', how='inner')
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

    result_alpha = pd.DataFrame(columns=['EventID', 'x', 'y', 'z', 'E', 'r'])
    result_alpha['EventID'] = TrigNum[cut][:,1]
    result_alpha['x'] = X[cut][:,1] / 1000
    result_alpha['y'] = Y[cut][:,1] / 1000
    result_alpha['z'] = Z[cut][:,1] / 1000
    result_alpha['E'] = E[cut][:,1]
    result_alpha['r'] = np.sqrt(result_alpha['x']**2 + result_alpha['y']**2 + result_alpha['z']**2)
    result_beta = pd.DataFrame(columns=['EventID', 'x', 'y', 'z', 'E', 'r'])
    result_beta['EventID'] = TrigNum[cut][:,0]
    result_beta['x'] = X[cut][:,0] / 1000
    result_beta['y'] = Y[cut][:,0] / 1000
    result_beta['z'] = Z[cut][:,0] / 1000
    result_beta['E'] = E[cut][:,0]
    result_beta['r'] = np.sqrt(result_beta['x']**2 + result_beta['y']**2 + result_beta['z']**2)
    result_bc = pd.concat([result_alpha, result_beta])
    print("done")
    return result_bc   

def charge_sum(data):
    gain = pd.DataFrame(columns=['ch', 'mus'])
    with h5py.File(args.ser,"r") as ipt:
        gain['ch'] = ipt['ser']['ch']
        gain['mus'] = ipt['ser']['mus']
    merged_df = pd.merge(data.iloc[:1000], gain, on='ch')
    merged_df['NPE'] = merged_df['q'] * merged_df['count'] / merged_df['mus'] / 2500
    result = merged_df.groupby('EventID')['NPE'].sum().reset_index()
    return result

def read_fsmp(inputfiles):
    print("read fsmp")
    result_fsmp = pd.DataFrame(columns=['EventID', 'NPE'])
    for f in range(len(inputfiles)):
        data = pq.read_table(inputfiles[f]).to_pandas()
        data.rename(columns={'eid': 'EventID'}, inplace=True)
        result_fsmp = pd.concat([result_fsmp, charge_sum(data)])
    result_fsmp['EventID'] = result_fsmp['EventID'].astype('int')
    print("done")
    return result_fsmp

# step = pd.merge(read_recon(args.step, "step"), eids[['EventID', 'particle']], on='EventID', how='inner')
# stack = pd.merge(read_recon(args.stack, "stack"), eids[['EventID', 'particle']], on='EventID', how='inner')
# bc = pd.merge(read_bc(args.bc), eids[['EventID', 'particle']], on='EventID', how='inner')
fsmp = pd.merge(read_fsmp(args.fsmp), eids[['EventID', 'particle']], on='EventID', how='inner')
breakpoint()
opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    breakpoint()
    # opt.create_dataset("step", data=step.to_records(index=False), **opts)
    #opt.create_dataset("stack", data=stack.to_records(index=False), **opts)
    #opt.create_dataset("bc", data=bc.to_records(index=False), **opts)
    opt.create_dataset("fsmp", data=fsmp.to_records(index=False), **opts)

