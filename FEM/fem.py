import numpy as np

from fealpy.mesh import TriangleMesh
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt


class Node:

    def __init__(self, x, y, serial_num):
        self.__x = x
        self.__y = y
        self.__serial_num = serial_num

    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y

    @property
    def sn(self):
        return self.__serial_num
    


class Cell:

    def __init__(self, nodes, cell):
        self.node_num, _ = nodes.shape
        to_measure = np.concatenate((np.ones((self.node_num, 1)), nodes), 
                                                                  axis=1)
        self.measure = np.linalg.det(to_measure)
        nodes = np.concatenate((nodes, cell.reshape((-1, 1))), axis=1)
        self.nodes = [Node(*node) for node in nodes]

    def b(self, i):
        return self.nodes[(i+1)%3].y-self.nodes[(i+2)%3].y

    def c(self, i):
        return self.nodes[(i+2)%3].x-self.nodes[(i+1)%3].x

    def get_Kmatrix(self, N):
        K = np.empty((self.node_num, self.node_num, 3))
        for i, nodei in enumerate(self.nodes):
            for j, nodej in enumerate(self.nodes):
                if i<=j:
                    K[i, j, 2] = (self.b(i)*self.b(j)+self.c(i)*self.c(j))/(4*self.measure)
                K[i, j, 0], K[i, j, 1] = nodei.sn, nodej.sn
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i>j:
                    K[i, j, 2] = K[j, i, 2]

        row = K[..., 0].flatten()
        col = K[..., 1].flatten()
        data = K[..., 2].flatten()
        K_matrix = csc_matrix((data, (row, col)), shape=(N, N))

        return K_matrix


def creat_mesh(init_node, init_cell, *, n, renturn_mesh=False):
    '''
    函数说明:网格生成函数

    Paramters:
        init_node - 初始网格节点
        init_cell - 初始节点单元
        n - 网格加密次数

    Return:
        node - 所有网格节点坐标
        cell - 所有节点单元
    '''
    mesh = TriangleMesh(init_node, init_cell)
    mesh.uniform_refine(n=n)
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    
    yield node
    yield cell
    if renturn_mesh: yield mesh


def get_K(node, cell):
    '''
    函数说明:生成总体系数矩阵

    Paramters:
        node - 网格节点
        cell - 节点单元

    Return:
        K - 总体系数矩阵
    '''
    N, _ = node.shape
    K = csc_matrix((N, N), dtype=np.int8)
    for c in cell:
        K += Cell(node[c], c).get_Kmatrix(N)
    
    return K.toarray()


def show_mesh(mesh, *, node=True,
                       cell=True,
                       edge=False,
                       show=True,
                       ):
    """
    函数说明:展示网格划分

    Paramters:
        mesh - 网格对象

    Return:
        None
    """
    fig = plt.figure()
    ax = fig.gca()
    mesh.add_plot(ax)
    if node: mesh.find_node(ax, showindex=True)
    if cell: mesh.find_cell(ax, showindex=True)
    if edge: mesh.find_edge(ax, showindex=True)
    if show: plt.show()
    