#!/usr/bin/env python3
'''
Prepare data for sim figure
'''
import pandas as pd
import argparse
import h5py
import numpy as np
import uproot

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help="output")
psr.add_argument('-i', dest='ipt', nargs="+", help="recon without time")
psr.add_argument('-w', dest='withtime', nargs="+", help="recon with time")
psr.add_argument('-t', dest='truth', nargs="+", help="truth")
args = psr.parse_args()

result_mcmc = pd.DataFrame(columns=['FileNo', 'EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'r', 'xy', 'std_xy', 'std_z'])
result_time = pd.DataFrame(columns=['FileNo', 'EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'r', 'xy', 'std_xy', 'std_z'])
result_truth = pd.DataFrame(columns=['FileNo', 'EventID', 'x', 'y', 'z', 'E', 'r', 'xy'])

for f in range(len(args.ipt)):
    with h5py.File(args.ipt[f],"r") as ipt:
        recon = pd.DataFrame(ipt['Recon'][:])
        recon['r'] = np.sqrt(recon['x']**2 + recon['y']**2 + recon['z']**2)
        recon['xy'] = recon['x']**2 + recon['y']**2
        recon_data = recon.groupby('EventID').mean().reset_index()
        std = recon.groupby('EventID').agg({'xy': 'std', 'z': 'std'})
        std = std.rename(columns={'xy': 'std_xy', 'z': 'std_z'})
        recon_data = pd.merge(recon_data, std, on='EventID', how='inner')
        recon_data.insert(0, 'FileNo', f)
        result_mcmc = pd.concat([result_mcmc, recon_data])  

for f in range(len(args.withtime)):
    with h5py.File(args.withtime[f],"r") as ipt:
        recon = pd.DataFrame(ipt['Recon'][:])
        recon['r'] = np.sqrt(recon['x']**2 + recon['y']**2 + recon['z']**2)
        recon['xy'] = recon['x']**2 + recon['y']**2
        recon_data = recon.groupby('EventID').mean().reset_index()
        std = recon.groupby('EventID').agg({'xy': 'std', 'z': 'std'})
        std = std.rename(columns={'xy': 'std_xy', 'z': 'std_z'})
        recon_data = pd.merge(recon_data, std, on='EventID', how='inner')
        recon_data.insert(0, 'FileNo', f)
        result_time = pd.concat([result_time, recon_data])

for f in range(len(args.truth)):
    with uproot.open(args.truth[f]) as ipt:
        info = ipt['SimTriggerInfo']
        PMTID = info['PEList.PMTId'].array()
        x = np.array(info['truthList.x'].array() / 1000).flatten()
        y = np.array(info['truthList.y'].array() / 1000).flatten()
        z = np.array(info['truthList.z'].array() / 1000).flatten()
        r = np.sqrt(x**2 + y**2 + z**2)
        xy = x**2 + y**2
        # 沿袭 dw 的规则，用 sid
        TriggerNo = np.array(info['TriggerNo'].array()).flatten()
        #TriggerNo = np.array(info['truthList.SegmentId'].array()).flatten()
        NPE = [len(sublist) for sublist in PMTID]
        truth = pd.DataFrame({"EventID":TriggerNo, "x":x, "y":y, "z":z, "E":NPE, "r":r, "xy":xy})
        truth.insert(0, 'FileNo', f)
        result_truth = pd.concat([result_truth, truth])

# 写入重建与真值
result_mcmc['EventID'] = result_mcmc['EventID'].astype('int')
result_time['EventID'] = result_time['EventID'].astype('int')
result_truth['EventID'] = result_truth['EventID'].astype('int')
result_mcmc['FileNo'] = result_mcmc['FileNo'].astype('int')
result_time['FileNo'] = result_time['FileNo'].astype('int')
result_truth['FileNo'] = result_truth['FileNo'].astype('int')
result_truth['E'] = result_truth['E'].astype('int')
opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("mcmc", data=result_mcmc.to_records(index=False), **opts)
    opt.create_dataset("time", data=result_time.to_records(index=False), **opts)
    opt.create_dataset("truth", data=result_truth.to_records(index=False), **opts)
