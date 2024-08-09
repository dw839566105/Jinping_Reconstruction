#!/usr/bin/env python3
'''
对顶点的分布取一个代表值: 暂为链 burn 后的均值
'''
import argparse
import h5py
import numpy as np
import pandas as pd
from DetectorConfig import shell

psr = argparse.ArgumentParser()
psr.add_argument("-i", dest="ipt", help="input recon file")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument("-s", dest="step", type=int, help="MC step")
psr.add_argument("-r", dest="rate", type=float, help="burn rate")
args = psr.parse_args()
rate = args.rate
step = args.step

def read_burn(file):
    '''
    burn 前 rate 的链，对分布取均值，归一化坐标加单位
    '''
    with h5py.File(file,"r") as ipt:
        # 结构化数组无法二元运算，转成 dataframe
        recon = pd.DataFrame(ipt['Recon'][:, int(rate * step):].reshape(-1))
        res = recon.groupby('EventID').mean().reset_index()
        res['x'] = res['x'].apply(lambda x: x * shell)
        res['y'] = res['y'].apply(lambda x: x * shell)
        res['z'] = res['z'].apply(lambda x: x * shell)
        res['r'] = np.sqrt(res['x'].values**2 + res['y'].values**2 + res['z'].values**2)
    return res

data = read_burn(args.ipt)
opts = {"compression": "gzip", "shuffle": True}
with h5py.File(args.opt, "w") as opt:
    opt.create_dataset("Recon", data=data.to_records(index=False), **opts)