#!/usr/bin/env python3
'''
Prepare data for sim figure
'''
import pandas as pd
import argparse
import h5py
import numpy as np
import uproot
from DetectorConfig import shell

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help="output")
psr.add_argument('-t', dest='truth', nargs="+", help="truth file")
psr.add_argument('-r', dest='recon', nargs="+", help="recon file")
psr.add_argument("-s", dest="step", type=int, help="MC step")
args = psr.parse_args()

def read_truth(inputfiles):
    result_truth = pd.DataFrame(columns=['FileNo', 'EventID', 'x', 'y', 'z', 'NPE', 'r', 'xy'])
    for f in range(len(inputfiles)):
        with uproot.open(inputfiles[f]) as ipt:
            info = ipt['SimTriggerInfo']
            PMTID = info['PEList.PMTId'].array()
            x = np.array(info['truthList.x'].array() / 1000).flatten()
            y = np.array(info['truthList.y'].array() / 1000).flatten()
            z = np.array(info['truthList.z'].array() / 1000).flatten()
            r = np.sqrt(x**2 + y**2 + z**2)
            xy = x**2 + y**2
            TriggerNo = np.array(info['TriggerNo'].array()).flatten()
            NPE = [len(sublist) for sublist in PMTID]
            truth = pd.DataFrame({"EventID":TriggerNo, "x":x, "y":y, "z":z, "NPE":NPE, "r":r, "xy":xy})
            truth.insert(0, 'FileNo', f)
            result_truth = pd.concat([result_truth, truth])
    result_truth['EventID'] = result_truth['EventID'].astype('int')
    result_truth['FileNo'] = result_truth['FileNo'].astype('int')
    result_truth['NPE'] = result_truth['NPE'].astype('int')
    return result_truth

def read_recon(inputfiles):
    result_mcmc = pd.DataFrame(columns=['EventID', 'step', 'x', 'y', 'z', 'E', 't', 'Likelihood', 'acceptz', 'acceptr', 'acceptE', 'acceptt', 'r'])
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
            recon_data.insert(0, 'FileNo', f)
            result_mcmc = pd.concat([result_mcmc, recon_data])
    result_mcmc['FileNo'] = result_mcmc['FileNo'].astype('int')
    result_mcmc['EventID'] = result_mcmc['EventID'].astype('int')
    return result_mcmc

truth = read_truth(args.truth)
recon = read_recon(args.recon)

opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("truth", data=truth.to_records(index=False), **opts)
    opt.create_dataset("recon", data=recon.to_records(index=False), **opts)
