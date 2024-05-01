#!/usr/bin/env python3
'''
Prepare data for sim figure
'''
import pandas as pd
import argparse
import h5py
import numpy as np
import uproot

shell = 0.65
psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help="output")
psr.add_argument('--truth', dest='truth', nargs="+", help="truth")
psr.add_argument('--stack', dest='stack', nargs="+", help="recon stack without time")
psr.add_argument('--stackt', dest='withtime', nargs="+", help="recon stack with time")
psr.add_argument('--step', dest='step', nargs="+", help="recon step without time")
psr.add_argument('--stept', dest='stept', nargs="+", help="recon step with time")
psr.add_argument('--bc', dest='bc', type=int, default=1, help="bc ON/OFF")
args = psr.parse_args()

def read_truth(inputfiles):
    result_truth = pd.DataFrame(columns=['FileNo', 'EventID', 'x', 'y', 'z', 'E', 'r', 'xy'])
    for f in range(inputfiles):
        with uproot.open(inputfiles[f]) as ipt:
            info = ipt['SimTriggerInfo']
            PMTID = info['PEList.PMTId'].array()
            x = np.array(info['truthList.x'].array() / 1000).flatten()
            y = np.array(info['truthList.y'].array() / 1000).flatten()
            z = np.array(info['truthList.z'].array() / 1000).flatten()
            r = np.sqrt(x**2 + y**2 + z**2)
            xy = x**2 + y**2
            # 沿袭 dw 的规则，用 sid
            TriggerNo = np.array(info['TriggerNo'].array()).flatten()
            # TriggerNo = np.array(info['truthList.SegmentId'].array()).flatten()
            NPE = [len(sublist) for sublist in PMTID]
            truth = pd.DataFrame({"EventID":TriggerNo, "x":x, "y":y, "z":z, "E":NPE, "r":r, "xy":xy})
            truth.insert(0, 'FileNo', f)
            result_truth = pd.concat([result_truth, truth])
    result_truth['EventID'] = result_truth['EventID'].astype('int')
    result_truth['FileNo'] = result_truth['FileNo'].astype('int')
    result_truth['E'] = result_truth['E'].astype('int')
    return result_truth

def read_recon(inputfiles):
    result_mcmc = pd.DataFrame(columns=['FileNo', 'EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'r', 'xy', 'std_xy', 'std_z'])
    for f in range(len(inputfiles)):
        with h5py.File(inputfiles[f],"r") as ipt:
            recon = pd.DataFrame(ipt['Recon'][:])
            recon['r'] = np.sqrt(recon['x']**2 + recon['y']**2 + recon['z']**2)
            recon['xy'] = recon['x']**2 + recon['y']**2
            recon_data = recon.groupby('EventID').mean().reset_index()
            std = recon.groupby('EventID').agg({'xy': 'std', 'z': 'std'})
            std = std.rename(columns={'xy': 'std_xy', 'z': 'std_z'})
            recon_data = pd.merge(recon_data, std, on='EventID', how='inner')
            recon_data.insert(0, 'FileNo', f)
            result_mcmc = pd.concat([result_mcmc, recon_data]) 
    result_mcmc['EventID'] = result_mcmc['EventID'].astype('int')
    result_mcmc['FileNo'] = result_mcmc['FileNo'].astype('int')
    return result_mcmc

def read_bc(inputfiles):
    result_bc = pd.DataFrame(columns=['FileNo', 'EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'r', 'xy'])
    for f in range(len(inputfiles)):
        with h5py.File(inputfiles[f],"r") as ipt:
            reconbc = pd.DataFrame(ipt['ReconBC'][:])
            reconbc[['x', 'y', 'z']] = reconbc[['x', 'y', 'z']].apply(lambda x: x * shell)
            reconbc['E'] = reconbc['E'].apply(lambda x: x / 2500 * 2 * 65 / 155)
            reconbc['r'] = np.sqrt(reconbc['x']**2 + reconbc['y']**2 + reconbc['z']**2)
            reconbc['xy'] = reconbc['x']**2 + reconbc['y']**2
            reconbc_data = reconbc.groupby('EventID').mean().reset_index()
            reconbc_data.insert(0, 'FileNo', f)
            result_bc = pd.concat([result_bc, reconbc_data])   
    result_bc['EventID'] = result_bc['EventID'].astype('int')   
    result_bc['FileNo'] = result_bc['FileNo'].astype('int')    
    return result_bc

truth = read_truth(args.truth)
stack = read_recon(args.stack)
stackt = read_recon(args.stackt)
step = read_recon(args.step)
stept = read_recon(args.stept)
if args.bc:
    bc = read_bc(args.step)

opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("truth", data=truth.to_records(index=False), **opts)
    opt.create_dataset("stack", data=stack.to_records(index=False), **opts)
    opt.create_dataset("stackt", data=stackt.to_records(index=False), **opts)
    opt.create_dataset("step", data=step.to_records(index=False), **opts)
    opt.create_dataset("stept", data=stept.to_records(index=False), **opts)
    if args.bc:
        opt.create_dataset("bc", data=bc.to_records(index=False), **opts)
