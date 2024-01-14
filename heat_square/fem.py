import numpy as np


def gen_grid(alpha: float, u0: float, u1: float, u2: float, nx=50, ny=50):
    # 定义网格和参数
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny

    # 初始化温度场
    u = np.zeros((nx + 1, ny + 1))
    u[:, :] = 1.0
    # 定义边界条件
    u[:, 0] = 1.0  # 左边界
    u[:, -1] = 1.0  # 右边界
    u[0, :] = 1.0  # 下边界
    u[-1, :] = -1.0  # 上边界
    u[-1, -1] = 0  # 角点修正
    u[-1, 1] = 0  # 角点修正
    # 定义时间步长和总时间
    dt = 0.01
    T = 30

    u_out = np.array([u])

    # 离散化方程
    r = alpha * dt / dx ** 2
    for t in np.arange(0, T, dt):
        un = u.copy()
        u[1:-1, 1:-1] = un[1:-1, 1:-1] + r * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) + r * (
                un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
        u_out = np.concatenate((u_out, [u]))
    return u_out


def grid_interpolate(u_out, new_point, nx=50, ny=50):
    # u_out 是第一个函数的输出 先用第一个函数跑出来结果再给第二个函数读取，new-point 是坐标的位置(x,y,t) x,y in [0,1] t in [0,30]
    dt = 0.01
    T = 30
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    time = np.linspace(0, T, int(T / dt) + 1)

    # 在一个新的点 (2.5, 2.5, 2.5) 进行插值

    dx = Lx / nx
    dy = Ly / ny
    xid = int(new_point[0] / dx)
    yid = int(new_point[1] / dy)
    xid_ = xid + 1
    yid_ = yid + 1
    tid = int(new_point[2] / dt)
    tid_ = tid + 1
    epsilon = (new_point[0] - x[xid]) / dx
    eta = (new_point[1] - y[yid]) / dy
    gamma = (new_point[2] - time[tid]) / dt
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


    # # 可视化结果
    # x = np.linspace(0, Lx, nx + 1)
    # y = np.linspace(0, Ly, ny + 1)
    # X, Y = np.meshgrid(x, y)
    #
    # plt.contourf(X, Y, u_out[1000], cmap='hot', levels=50)
    # plt.colorbar()
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('2D Heat Conduction')
    # plt.show()
    value = grid_interpolate(u_out, [0.33, 0.33, 21.005])
    print(value)
    # # 将空间和时间坐标展平，以适应插值函数的输入格式
    # points = np.array(np.meshgrid(x, y, time)).T.reshape(-1, 3)
    # print(points.shape)
    # values = u_out.flatten()
    # print(values.shape)
    # # 使用三维插值
    # interpolated_temperature = griddata(points, values, new_point, method='linear')
    #
    # print(f'Temperature at (2.5, 2.5, 2.5): {interpolated_temperature[0]}')
