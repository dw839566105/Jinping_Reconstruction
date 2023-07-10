import numpy as np


import argparse

psr = argparse.ArgumentParser()
psr.add_argument("ipt", type=str, help="Input detsim h5 file")
psr.add_argument("-o", "--opt", type=str, dest="opt", help="Output concat h5 file")
psr.add_argument("--pmt", type=str, dest="pmt", default='./PMT.txt', help="Output concat h5 file")
args = psr.parse_args()

geo_card = np.loadtxt(args.pmt)
npm = len(geo_card)

import h5py as h5
import pandas as pd

ipt = h5.File(args.ipt, "r")

def __thetas(xs, ys, zs, pmt_ids, pmt_poss):
    vertex_poss = np.array([xs, ys, zs]).T
    vertex_poss_norm = np.linalg.norm(vertex_poss, axis=1)
    vertex_poss_norm = vertex_poss_norm.reshape(len(vertex_poss_norm), 1)
    vertex_poss /= vertex_poss_norm
    pmt_pos_by_ids = pmt_poss[pmt_ids]
    pmt_pos_by_ids_norm = np.linalg.norm(pmt_pos_by_ids, axis=1)
    pmt_pos_by_ids_norm = pmt_pos_by_ids_norm.reshape(len(pmt_pos_by_ids_norm), 1)
    pmt_pos_by_ids /= pmt_pos_by_ids_norm
    thetas = np.arccos(
        np.clip(np.einsum("ij, ij -> i", vertex_poss, pmt_pos_by_ids), -1, 1)
    )
    return thetas


truth = ipt["SimTruth/SimTruth"][()]
cvt = ipt["SimTriggerInfo/TruthList"][()]
pe = ipt["SimTriggerInfo/PEList"][()]
df_truth = pd.DataFrame(truth)
df_cvt = pd.DataFrame(cvt)
df_pe = pd.DataFrame(pe)

with h5.File(args.opt, "w") as opt:
    # nonhit
    vertices = opt.create_dataset(
        "Vertices",
        shape=(len(cvt) * len(geo_card),),
        dtype=[("r", "<f4"), ("theta", "<f4")],
    )

    df_nonhit = df_cvt.merge(df_truth, how='inner', on='SegmentId')
    parx = df_nonhit["x"]
    pary = df_nonhit["y"]
    parz = df_nonhit["z"]
    
    xsr = parx.repeat(npm)
    ysr = pary.repeat(npm)
    zsr = parz.repeat(npm)
    pmt_ids = np.tile(np.arange(npm), len(parx))
    thetas = __thetas(xsr, ysr, zsr, pmt_ids, geo_card)
    rsr = np.sqrt(xsr ** 2 + ysr ** 2 + zsr ** 2)
    vertices["r"] = rsr
    vertices["theta"] = thetas

    # hit
    concat = opt.create_dataset(
        "Concat",
        shape=(len(pe),),
        dtype=[("EId", "<i"), ("CId", "<i"), ("r", "<f4"), ("theta", "<f4"), ("t", "<f4")],
    )
    df_hit = df_pe.merge(df_truth, how='inner', on='SegmentId')
    xs = df_hit["x"]
    ys = df_hit["y"]
    zs = df_hit["z"]
    rs = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    pmt_ids = df_hit["PMTId"]
    thetas = __thetas(xs, ys, zs, pmt_ids, geo_card)
    ts = df_hit["PulseTime"]
    concat["EId"] = df_hit['TriggerNo']
    concat["CId"] = df_hit["PMTId"]
    concat["r"] = rs
    concat["theta"] = thetas
    concat["t"] = ts
