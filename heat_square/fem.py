import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定义网格和参数
dt = 0.0001
T = 30

nx, ny = 50, 50
Lx, Ly = 1.0, 1.0

dx, dy = Lx / nx, Ly / ny

x = np.linspace(0, Lx, nx + 1)
y = np.linspace(0, Ly, ny + 1)
times = np.linspace(0, T, int(T / dt) + 1)

def gen_grid(alpha):
    # 初始化温度场
    u = np.zeros((nx + 1, ny + 1))
    u[:, :] = 0
    # 定义边界条件
    u[:, 0] = 0  # 左边界
    u[:, -1] = 0  # 右边界
    u[0, :] = 0  # 下边界
    u[-1, :] = 1.0  # 上边界
    u[-1, -1] = 0.5  # 角点修正
    u[-1, 0] = 0.5  # 角点修正

    u =u.astype(np.float64)
    u_out = np.array([u])

    # 离散化方程
    r = alpha * dt / dx ** 2
    print(r)
    count =0
    for t in times:
        count += 1
        un = u.copy()
        u[1:-1, 1:-1] = un[1:-1, 1:-1] + r * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) + r * (
                un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
        if count % 100 == 0:
            u_out = np.concatenate((u_out, [u]))
        if count % 1000 == 0:
            print(count/1000)
    return u_out


def grid_interpolate(u_out, new_point):
    # u_out 是第一个函数的输出 先用第一个函数跑出来结果再给第二个函数读取，new-point 是坐标的位置(x,y,t) x,y in [0,1] t in [0,30]
    xid = int(new_point[0] / dx)
    yid = int(new_point[1] / dy)
    xid_ = xid + 1
    yid_ = yid + 1
    tid = int(new_point[2] / dt /100)
    tid_ = tid + 1
    if xid >= len(x)-1:
        xid_ = xid
    if yid >= len(y)-1:
        yid_ = yid
    if tid >= len(times)-1:
        tid_ = tid

    epsilon = (new_point[0] - x[xid]) / dx
    eta = (new_point[1] - y[yid]) / dy
    gamma = (new_point[2] - times[tid]) / dt
    print(eta, epsilon)
    v1, v2, v3, v4 = u_out[tid, xid, yid], u_out[tid, xid, yid_], u_out[tid, xid_, yid], u_out[tid, xid, yid_]
    value = (v1 * (1 - epsilon) + v2 * (1 - epsilon)) * (1 - eta) + (v3 * epsilon + v4 * epsilon) * eta
    v1, v2, v3, v4 = u_out[tid_, xid, yid], u_out[tid_, xid, yid_], u_out[tid_, xid_, yid], u_out[tid_, xid, yid_]
    value_ = (v1 * (1 - epsilon) + v2 * (1 - epsilon)) * (1 - eta) + (v3 * epsilon + v4 * epsilon) * eta
    value = value * (1 - gamma) + value_ * (gamma)
    return value

if __name__ == '__main__':
    u_out = gen_grid()
    print(u_out.shape)

    # 创建初始图像
    fig, ax = plt.subplots()
    heatmap = ax.imshow(u_out[0], cmap='plasma', origin='lower', interpolation='nearest')
    ax.set_title('Temperature Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(heatmap)


    # 更新图像的函数
    def update(frame):
        heatmap.set_array(u_out[frame])
        ax.set_title(f'Temperature Distribution at Time Point {frame}')
        return heatmap,

    print(len(times))

    # 创建动画
    animation = FuncAnimation(fig, update, frames=int(len(times) / 100), interval=10, blit=False)

    # 显示动画
    plt.show()

    value = grid_interpolate(u_out, [1.0, 1.0, 1])
    print(value)
    print('end')