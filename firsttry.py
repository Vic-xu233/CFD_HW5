import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_position, minimum_position

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#定义网格
N=136                  # 网格数
L = 1.0               # 区域尺寸
nu = 0.001            # 黏性系数
dx = L / (N - 1)      # 空间步长
dy = dx
dt = 0.01            # 时间步长
max_iter = 20000      # 最大迭代次数
threshold = 1e-7      # 收敛阈值

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
            if iteration > 2 and iteration % 100 == 0:
                print(f"SOR收敛，迭代次数 = {iteration+1}，误差 = {error:.2e}，松弛因子 ω = {w}")
            break
def boundary_conditions(omega, psi, delta_x, u_lid):
    omega[-1,:] = -2 * psi[:, -2] / delta_x**2 - 2 * u_lid / delta_x  # Top wall (moving lid)
    omega[0,:] = -2 * psi[:, 1] / delta_x**2  # Bottom wall
    omega[:, 0] = -2 * psi[1, :] / delta_x**2  # Left wall
    omega[:, -1] = -2 * psi[-2, :] / delta_x**2  # Right wall
    return omega
def apply_woods_boundary(omega, psi, dx, u_top):
    """
    使用 Woods 壁涡量公式更新边界涡量：
    - 上壁 (y=1)：使用公式 6.18，包含 u_top, ∂²u/∂x²
    - 下、左、右：使用公式 6.19（u=0 情况）
    """
    N = psi.shape[0]
    # ---------- 上壁 (y=1)：使用公式 (6.18) ----------
    i = -1       # 上壁索引
    i1 = -2      # 内部点
    # 计算 ∂²u/∂x²：中心差分
    d2udx2 = (u_top[2:] - 2 * u_top[1:-1] + u_top[:-2]) / dx**2
    omega[i, 1:-1] = (
        -0.5 * omega[i1, 1:-1]
        - (3 / dx**2) * (psi[i1, 1:-1] - psi[i, 1:-1])
        - (3 / dx) * u_top[1:-1]
        + (dx / 2) * d2udx2
    )
    # ---------- 下壁 (y=0)：使用简化式 (6.19) ----------
    omega[0, 1:-1] = (
        -0.5 * omega[1, 1:-1]
        - (3 / dx**2) * (psi[1, 1:-1] - psi[0, 1:-1])
    )
    # ---------- 左壁 (x=0)：使用简化式 (6.19) ----------
    omega[1:-1, 0] = (
        -0.5 * omega[1:-1, 1]
        - (3 / dx**2) * (psi[1:-1, 1] - psi[1:-1, 0])
    )
    # ---------- 右壁 (x=1)：使用简化式 (6.19) ----------
    omega[1:-1, -1] = (
        -0.5 * omega[1:-1, -2]
        - (3 / dx**2) * (psi[1:-1, -2] - psi[1:-1, -1])
    )
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
    omega=apply_woods_boundary(omega, psi, dx, u_top)
    update_psi(psi, omega,dx,dy)
    err = np.max(np.abs(omega - omega_old))
    if np.isnan(omega).any():
        print(f"NaN出现于迭代 {it}")
        break
    if it % 50 == 0:
        print(f"Iter {it}, omega_min={np.min(omega):.2f}, error={err:.2e}")
    if err < threshold:
        print(f"收敛于迭代 {it}")
        break

velocity_magnitude = np.sqrt(u_vel**2 + v_vel**2)

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.contourf(X, Y, velocity_magnitude, levels=100, cmap='jet')
plt.colorbar(label='速度大小')
plt.title('速度场分布')

plt.subplot(122)
plt.contour(X, Y, psi, levels=60, colors='k', linewidths=0.5)
plt.streamplot(X, Y, u_vel, v_vel, density=4, color='red')
plt.title('流线图')
plt.tight_layout()
plt.savefig('速度场和流线.png', dpi=300)
plt.show()
# 1. 垂直中线上的速度剖面（x = L/2）
# 取垂直中线（x=L/2）上的速度剖面
mid_x_idx = N // 2  # 水平方向中间的列索引
u_profile = u_vel[:, mid_x_idx]  # 沿y方向的u分量
plt.figure(figsize=(6,4))
plt.plot(y, u_profile, 'b-', lw=2)
plt.xlabel('y 位置')
plt.ylabel('u 速度 (x=L/2)')
plt.title('垂直中线上的速度剖面')
plt.grid(True)
plt.savefig('vertical_centerline_velocity_profile.png', dpi=300)
plt.close()
mid_y_idx = N // 2  # 垂直方向中间的行索引
v_profile = v_vel[mid_y_idx, :]  # 沿x方向的v分量
plt.figure(figsize=(6,4))
plt.plot(x, v_profile, 'r-', lw=2)
plt.xlabel('x 位置')
plt.ylabel('v 速度 (y=L/2)')
plt.title('水平中线上的速度剖面')
plt.grid(True)
plt.savefig('horizontal_centerline_velocity_profile.png', dpi=300)
plt.close()
# ========================
# 2. 主涡涡心、流函数极值位置
psi_max_pos = np.unravel_index(np.argmax(psi, axis=None), psi.shape)
psi_min_pos = np.unravel_index(np.argmin(psi, axis=None), psi.shape)
print(f"主涡中心（psi最大）物理坐标: x={x_max:.4f}, y={y_max:.4f}, ψ={psi[i_max, j_max]:.4f}")
print(f"主涡中心（psi最小）物理坐标: x={x_min:.4f}, y={y_min:.4f}, ψ={psi[i_min, j_min]:.4f}")
# 可视化极值点
plt.figure(figsize=(7,5))
plt.contourf(X, Y, psi, levels=50, cmap='viridis')
plt.colorbar(label='流函数ψ')
plt.scatter(X[psi_max_pos], Y[psi_max_pos], color='r', s=80, marker='*', label='ψ最大')
plt.scatter(X[psi_min_pos], Y[psi_min_pos], color='b', s=80, marker='*', label='ψ最小(主涡心)')
plt.legend()
plt.title('流函数及主涡涡心位置')
plt.savefig('psi_extrema.png', dpi=300)
plt.close()

# ========================
# 3. 角附近二次涡定位（可选，方法1：寻找四角一定范围内的极值）
corner_size = N // 8  # 取边角10%区域
corners = [
    psi[0:corner_size, 0:corner_size],
    psi[0:corner_size, -corner_size:],
    psi[-corner_size:, 0:corner_size],
    psi[-corner_size:, -corner_size:]
]
corner_names = ['左下', '右下', '左上', '右上']

for idx, region in enumerate(corners):
    # 找极大极小值
    min_pos = minimum_position(region)
    max_pos = maximum_position(region)
    # 全局索引
    min_global = (min_pos[0] + (0 if idx < 2 else N-corner_size),
                  min_pos[1] + (0 if idx % 2 == 0 else N-corner_size))
    max_global = (max_pos[0] + (0 if idx < 2 else N-corner_size),
                  max_pos[1] + (0 if idx % 2 == 0 else N-corner_size))
    print(f"{corner_names[idx]}角: ψ最小 at {min_global}, 值={psi[min_global]:.4f}")
    print(f"{corner_names[idx]}角: ψ最大 at {max_global}, 值={psi[max_global]:.4f}")

# ========================

# 总结
print('已保存图片：v_central_profile.png, psi_extrema.png')