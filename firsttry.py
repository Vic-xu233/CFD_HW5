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
dt = 0.01            # 时间步长
max_iter = 15000      # 最大迭代次数
threshold = 1e-8      # 收敛阈值

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
def compute_velocity(psi, u, v):
    nx, ny = psi.shape
    
    # 计算内部点的速度场
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u[i, j] = (psi[i, j+1] - psi[i, j-1]) / (2 * dy)  # u = ∂ψ/∂y
            v[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2 * dx)  # v = -∂ψ/∂x
    
    # 设置顶盖边界条件（假设顶盖在最后一行）
    for j in range(ny):
        u[-1, j] = u_top[j]  # 直接赋值为给定的顶盖速度分布
    
    return u, v
def update_psi(psi, omega, dx, dy, max_iter=10000, threshold=1e-6, w=1.8):
    beta2 = (dx / dy) ** 2
    coeff = 1 / (2 * (1 + beta2))
    
    for iteration in range(max_iter):
        psi_old = psi.copy()
        
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                residual = (
                    psi[i+1, j] + psi[i-1, j] +
                    beta2 * (psi[i, j+1] + psi[i, j-1]) +
                    dx**2 * omega[i, j]
                ) * coeff - psi[i, j]
                psi[i, j] += w * residual
        # 设置边界条件
        psi[0, :] = 0
        psi[-1, :] = 0
        psi[:, 0] = 0
        psi[:, -1] = 0

        # 收敛判断
        error = np.max(np.abs(psi - psi_old))
        if error < threshold:
            if iteration % 100 == 0:
                print(f"SOR收敛，迭代次数 = {iteration+1}，误差 = {error:.2e}，松弛因子 ω = {w}")
            break
def boundary_conditions(omega, psi, delta_x, u_lid):
    omega[-1,:] = -2 * psi[:, -2] / delta_x**2 - 2 * u_lid / delta_x  # Top wall (moving lid)
    omega[0,:] = -2 * psi[:, 1] / delta_x**2  # Bottom wall
    omega[:, 0] = -2 * psi[1, :] / delta_x**2  # Left wall
    omega[:, -1] = -2 * psi[-2, :] / delta_x**2  # Right wall
    return omega

for it in range(max_iter):
    omega_old = omega.copy()
    u_vel, v_vel = compute_velocity(psi,u_vel,v_vel)
    for i in range(N):
        for j in range(N):
            if i == 0 or j == 0 or i == N-1 or j == N-1:
                continue
            else:
                dwdx = (omega_old[i+1, j] - omega_old[i-1, j]) / (2*dx)
                dwdy = (omega_old[i, j+1] - omega_old[i, j-1]) / (2*dy)
                convective_term = (u_vel[i, j] * dwdx + v_vel[i, j] * dwdy)
                omega[i, j] = (nu*(omega_old[i+1, j] + omega_old[i-1, j] +
                                   omega_old[i, j+1] + omega_old[i, j-1] - 4*omega_old[i, j]) / (dx*dx)
                + convective_term) * dt+ omega_old[i, j]
    omega=boundary_conditions(omega, psi, dx, u_top)
    update_psi(psi, omega,dx,dy)
    err = np.max(np.abs(omega - omega_old))
    if np.isnan(omega).any():
        print(f"NaN出现于迭代 {it}")
        break
    if it % 100 == 0:
        print(f"Iter {it}, omega_min={np.min(omega):.2f}, error={err:.2e}")
    if err < threshold:
        print(f"收敛于迭代 {it}")
        break

velocity_magnitude = np.sqrt(u_vel**2 + v_vel**2)

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.contourf(X, Y, velocity_magnitude, levels=50, cmap='jet')
plt.colorbar(label='速度大小')
plt.title('速度场分布')

plt.subplot(122)
plt.contour(X, Y, psi, levels=30, colors='k', linewidths=0.5)
plt.streamplot(X, Y, u_vel, v_vel, density=2, color='white')
plt.title('流线图')
plt.tight_layout()
plt.show()