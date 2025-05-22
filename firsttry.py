import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#定义网格
N=50                  # 网格数
L = 1.0               # 区域尺寸
nu = 0.001            # 黏性系数
dx = L / (N - 1)      # 空间步长
dy = dx
dt = 0.001            # 时间步长
max_iter = 10000      # 最大迭代次数
threshold = 1e-6      # 收敛阈值

