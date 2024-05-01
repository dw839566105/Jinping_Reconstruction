#!/usr/bin/env python3
'''
重建所需文件读入
'''
import pandas as pd
import pyarrow.parquet as pq
import h5py
import numpy as np
import tables
from DetectorConfig import chs
import sys
sys.path.append('../')
from FSMP.fsmp_reader import FSMPreader, TRIALS

def ReadFile(file, sparsify, data_mode, time_mode, entries):
    '''
    读入待重建事例文件
    file: 波形分析文件或模拟真值
    sparsify: FSMP 的 sparsify 文件
    data_mode: sim/raw
    time_mode: ON/OFF
    entries: 单文件重建事例数
    '''
    if data_mode == "sim":
        return ReadPETruth(file, entries)
    else:
        return Readfsmp(file, sparsify, entries)

def ReadPETruth(file, entries):
    '''
    读入 PEt 真值
    '''
    data = pd.read_hdf(file, 'PEt')
    if entries != 0:
        data = data.query(f'eid < {entries}')
    return data

def Readfsmp(file, sparsify, entries):
    '''
    读入 FSMP 分析结果 fsmp, 并和 mu0 merge到一起
    '''
    data = pq.read_table(file, filters=[("ch", "in", list(range(chs)))]).to_pandas()
    if entries != 0:
        eid_list = data['eid'].unique()[:entries]
        data = data[data['eid'].isin(eid_list)]
        data = data[['eid', 'ch', 'offset', 'step', 's0', 'nu_lc', 'count']]
        data['s0'] = data['s0'].astype('int16')
        # 剔除初值
        data = data[data['count'] != 0]
    # 从 sparsify 读入 mu0
    with h5py.File(sparsify, "r") as f:
        mu0 = pd.DataFrame(f['index'][:])
        mu0['mu0'] = f['mu0'][:]
        if entries != 0:
            mu0 = mu0[mu0['eid'].isin(eid_list)]
        mu0 = mu0[['eid', 'ch', 'offset', 'mu0']]
    return pd.merge(data, mu0, on=['eid', 'ch','offset'], how='left')

def ReadPMT(file):
    '''
    读入 PMT 位置文件
    '''
    return np.loadtxt(file)

def ReadPEchain(sparsify, file):
    '''
    重新生成 PEt
    (待修改)
    '''
    for (eid, ch, offset), s0max, chain in FSMPreader(
        "/junofs/O/sparsify/0317/094422_1001/3.h5",
        "/junofs/O/fsmp/0317/094422_1001/3.pq").flat_iter():
    PEt = np.empty((5000 - 2500) * s0max, dtype=[("PEt", "f4"), ("count", "u2")])
    s_PEt = 0
    for s_x, e_x, s0, count in chain:
        PEt["PEt"][s_PEt:s_PEt+s0] = s_x[:s0]
        PEt["count"][s_PEt:s_PEt+s0] = count
        s_PEt += s0
    PEt[:s_PEt]["PEt"] += offset
    PEt = PEt[:s_PEt]
    return PEchain

class DataType(tables.IsDescription):
    '''
    重建数据储存类型
    '''
    EventID = tables.Int64Col(pos=0)    # EventNo
    step = tables.Int64Col(pos=1)       # wave step
    x = tables.Float32Col(pos=2)        # x position
    y = tables.Float32Col(pos=3)        # y position
    z = tables.Float32Col(pos=4)        # z position
    E = tables.Float32Col(pos=5)        # Energy
    t = tables.Float32Col(pos=6)        # time
    Likelihood = tables.Float64Col(pos=7)
    acceptz = tables.Float32Col(pos=8)
    acceptr = tables.Float32Col(pos=9)
    acceptE = tables.Float32Col(pos=10)
    acceptt = tables.Float32Col(pos=11)