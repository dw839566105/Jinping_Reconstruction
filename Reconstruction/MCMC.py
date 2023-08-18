import numpy as np

# 探测器参数
r_max = 0.1 # 单步距离晃动限额
T_max = 2 # 单步时间晃动限额 / ns

# 随机晃动一步
def perturbation(event):
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(-r_max, r_max)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    t = np.random.uniform(-T_max, T_max)
    return event + [x, y, z, t]
