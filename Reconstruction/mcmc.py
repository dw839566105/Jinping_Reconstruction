#!/usr/bin/env python3
'''
对变量采样，边界检查，重建初值
'''
import numpy as np
from config import *

def Perturb_energy(vertex, u):
    '''
    对 E 进行随机晃动
    '''
    E = E_max * (2 * u - 1)
    return vertex + [0, 0, 0, E, 0]

def Perturb_pos(vertex, u, r_max_E):
    '''
    对 (x,,y,z,t) 进行随机晃动
    '''
    cos_theta =  (2 * u[0] - 1)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = 2 * np.pi * u[1]
    r = r_max_E * u[2]
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    return vertex + [x, y, z, 0, 0]

def Perturb_zi(zis, cumul, vertex, expect, u_int, u):
    '''
    对 zi 进行抽样
    zis: FSMP 返回的某通道上的 Z
    cumul: 上一次抽样 z 的 cumulation
    u_int: 本次抽样 z 的 cumulation
    u: 0-1 间的随机数
    '''
    log_ratio = (zis['s0'].values[u_int] - zis['s0'].values[cumul]) * np.log(expect * vertex[3] / E0) + zis['nu_lc'].values[cumul] - zis['nu_lc'].values[u_int]
    if log_ratio > np.log(u):
        return u_int, zis['s0'].values[u_int]
    else:
        return cumul, zis['s0'].values[cumul]







