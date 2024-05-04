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

def Perturb_posT(vertex, u, r_max_E, time_mode):
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
    t = T_max * (2 * u[3] - 1)
    return vertex + [x, y, z, 0, t]

