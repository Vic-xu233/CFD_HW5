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

x=np.linspace(0, L, N)
y=np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)
u_top = np.sin(np.pi * x)**2  # 顶部速度分布

psi=np.zeros((N, N))  # 流函数
omega=np.zeros((N, N))  # 涡量  
u_vel=np.zeros((N, N))  # u 速度
v_vel=np.zeros((N, N))  # v 速度
# ----------------------
# 辅助函数定义
# ----------------------
def compute_velocity(psi,u,v):
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy)
    u[-1, :] = u_top
    return u, v
def update_psi(psi, omega):
    # 使用SOR方法更新流函数
    w=1.8
    beta=dx/dy
    for iteration in range(max_iter):
        psi_old = psi.copy() 
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T_new = (1 / (2 * (1 + beta**2))) * (
                    psi_old[i+1, j] + T[i-1, j] + beta**2 * (psi_old[i, j+1] + psi_old[i, j-1])
                )
                T[i, j] += omega * (T_new - T[i, j])
        # 边界条件
        psi[-1, :] = 0
        psi[:, 0] = psi[:, -1] = psi[0, :] = 0
        error = np.max(np.abs(T - T_old))
        if error < threshold:
            print(f"经过{iteration+1} 次迭代收敛，松弛因子 ω = {omega}")
            break
def boundary_conditions(omega, psi, delta_x, u_lid):
    omega[:, -1] = -2 * psi[:, -2] / delta_x**2 - 2 * u_lid / delta_x  # Top wall (moving lid)
    omega[:, 0] = -2 * psi[:, 1] / delta_x**2  # Bottom wall
    omega[0, :] = -2 * psi[1, :] / delta_x**2  # Left wall
    omega[-1, :] = -2 * psi[-2, :] / delta_x**2  # Right wall
    return omega

for it in range(max_iter):
    omega_old = omega.copy()
    u_vel, v_vel = compute_velocity(psi,u_vel,v_vel)
    for i in range(N):
        for j in range(N):
            if i == 0 or j == 0 or i == N-1 or j == N-1:
                continue
            else:
                dwdx = (omega[i+1, j] - omega[i-1, j]) / (2*dx)
                dwdy = (omega[i, j+1] - omega[i, j-1]) / (2*dy)
                convective_term = (u_vel[i, j] * dwdx + v_vel[i, j] * dwdy)
                omega[i, j] = (nu*(omega_old[i+1, j] + omega_old[i-1, j] +
                                   omega_old[i, j+1] + omega_old[i, j-1] - 4*omega_old[i, j]) / (dx*dx)
                + convective_term) * dt+ omega_old[i, j]
    omega=boundary_conditions(omega, psi, dx, u_top)
    update_psi(psi, omega)

    