import numpy as np

from matplotlib import pyplot as plt
from fem import creat_mesh, get_K, show_mesh


def p173_1(n):
    # 节点坐标
    init_node = np.array([(5, 5),
                          (5, 0),
                          (3, 0),
                          (1.5, 1.5)],
                          dtype=np.float64
                          )
    # 构成三角形的三个点构成的元组
    init_cell = np.array([(0, 1, 2),
                          (0, 2, 3)],
                          dtype=np.int_
                          )
    cm = creat_mesh(init_node, init_cell,
                    n=n,
                    renturn_mesh=True,
                    )
    node = next(cm)
    cell = next(cm)

    mesh = next(cm)
    show_mesh(mesh, cell=False, show=False)

    K = get_K(node, cell)
    N, _ = node.shape

    phi = np.full((N, 1), 0.5)
    w = 0.1
    for _ in range(333):
        for i, p in enumerate(phi):
            if node[i, 0] == 5:
                phi[i, 0] = 0
                continue
            if abs(node[i, 0])+abs(node[i, 1]) == 3:
                phi[i, 0] = 1
                continue
            phi_ = np.delete(phi, i)
            Ki_ = np.delete(K[i].reshape((1, -1)), i)
            p = np.sum(phi_*Ki_)
            phi[i, 0] = phi[i, 0]+w*(-p/K[i, i]-phi[i, 0])

    x = np.linspace(-3, 3, 60)
    y = x
    xx, yy = np.meshgrid(x, y)
    XX, YY = (xx-yy)/2, (xx+yy)/2
    ZZ = np.ones_like(XX)
    ZZ[0, 0] = 0

    X, Y = node[:, 0], node[:, 1]

    cmap = plt.cm.get_cmap('rainbow')
    norm = plt.Normalize(0, 1)
    # maxcolor = cmap(nol(1.))
    fig1 = plt.figure()
    f1_ax = fig1.add_subplot(1, 1, 1)
    f1_ax.set_aspect('equal')
    trf = f1_ax.tricontourf(X, Y, cell, phi[:, 0], cmap=cmap)
    f1_ax.tricontourf(-X, Y, cell, phi[:, 0], cmap=cmap)
    f1_ax.tricontourf(X, -Y, cell, phi[:, 0], cmap=cmap)
    f1_ax.tricontourf(-X, -Y, cell, phi[:, 0], cmap=cmap)
    f1_ax.tricontourf(Y, X, cell, phi[:, 0], cmap=cmap)
    f1_ax.tricontourf(-Y, X, cell, phi[:, 0], cmap=cmap)
    f1_ax.tricontourf(Y, -X, cell, phi[:, 0], cmap=cmap)
    f1_ax.tricontourf(-Y, -X, cell, phi[:, 0], cmap=cmap)
    f1_ax.contourf(XX, YY, ZZ, cmap=cmap, norm=norm)
    fig1.colorbar(trf)

    fig2 = plt.figure()
    f2_ax = fig2.add_subplot(1, 1, 1, projection='3d')
    trp = f2_ax.plot_trisurf(X, Y, cell, phi[:, 0], cmap=cmap)
    f2_ax.plot_trisurf(-X, Y, cell, phi[:, 0], cmap=cmap)
    f2_ax.plot_trisurf(X, -Y, cell, phi[:, 0], cmap=cmap)
    f2_ax.plot_trisurf(-X, -Y, cell, phi[:, 0], cmap=cmap)
    f2_ax.plot_trisurf(Y, X, cell, phi[:, 0], cmap=cmap)
    f2_ax.plot_trisurf(-Y, X, cell, phi[:, 0], cmap=cmap)
    f2_ax.plot_trisurf(Y, -X, cell, phi[:, 0], cmap=cmap)
    f2_ax.plot_trisurf(-Y, -X, cell, phi[:, 0], cmap=cmap)
    f2_ax.plot_surface(XX, YY, ZZ, cmap=cmap, norm=norm)
    fig2.colorbar(trp)

    plt.show()


if __name__ == '__main__':
    p173_1(3)
