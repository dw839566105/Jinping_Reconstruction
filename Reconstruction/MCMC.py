import numpy as np

# 探测器参数
r_max = 0.1 # 单步距离晃动限额
T_max = 2 # 单步时间晃动限额 / ns

# 随机晃动一步
def perturbation(event):
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(-r_max, r_max)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    t = np.random.uniform(-T_max, T_max)
    return event + [x, y, z, t]
