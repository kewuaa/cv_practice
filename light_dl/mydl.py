from backend import backend as np
from operations import GraphNode
from gradfunc import expfunc, logfunc, softmaxfunc
from math import log as mlog


def exp(value):
    if not isinstance(value, GraphNode):
        value = GraphNode(value)
    new_tensor = np.exp(value.tensor)
    new_node = GraphNode(new_tensor)
    new_node.requires_grad = value.requires_grad
    new_node.gradfunc = expfunc
    new_node.is_leaf = False
    new_node.subnode_l = value
    value.father = new_node
    new_node.operation = 'exp'
    return new_node


def log(value, base=None):
    if not isinstance(value, GraphNode):
        value = GraphNode(value)
    new_tensor = np.log(value.tensor)
    if base is not None:
        new_tensor /= mlog(base)
    new_node = GraphNode(new_tensor)
    new_node.requires_grad = value.requires_grad
    new_node.gradfunc = logfunc
    new_node.is_leaf = False
    new_node.subnode_l = value
    new_node.subnode_r = base
    value.father = new_node
    new_node.operation = 'log'
    return new_node


def softmax(X):
    exp_x = np.exp(X)
    partition = exp_x.sum(axis=1)
    partition = np.expand_dims(partition, axis=1)
    return np.ma.true_divide(exp_x, partition).filled(1e+10)


def softmax_loss(X, Y):
    '''
    X一行为一组
    Y为一列标签
    '''
    if not isinstance(X, GraphNode):
        X = GraphNode(X)
    Y = np.asarray(Y).ravel()
    row, col = zip(*enumerate(Y))
    x = X.tensor-np.expand_dims(X.tensor.max(axis=1), axis=1)
    logx = np.log(np.exp(x.sum(axis=1)))
    # softmax_x = softmax(X.tensor)
    # new_tensor = -(np.ma.log(softmax_x[row, col]).filled(0)).sum()
    new_tensor = -(x[row, col]-logx).sum()
    new_node = GraphNode(new_tensor)
    new_node.requires_grad = X.requires_grad
    new_node.gradfunc = softmaxfunc
    new_node.is_leaf = False
    new_node.subnode_l = X
    new_node.subnode_r = softmax(x)-np.eye(X.tensor.shape[1])[Y]
    X.father = new_node
    new_node.operation = 'softmaxloss'
    return new_node


def arange(start=0, stop=None, step=1, *, requires_grad=False):
    if stop is None:
        start, stop = 0, start
    array = np.arange(start, stop, step)
    return GraphNode(array, requires_grad=requires_grad)


def ones(shape, *, requires_grad=False):
    ones = np.ones(shape)
    return GraphNode(ones, requires_grad=requires_grad)


def ones_like(array_like, *, requires_grad=False):
    if isinstance(array_like, GraphNode):
        array_like = array_like.tensor
    ones = np.ones_like(array_like)
    return GraphNode(ones, requires_grad=requires_grad)


def zeros(shape, *, requires_grad=False):
    zeros = np.zeros(shape)
    return GraphNode(zeros, requires_grad=requires_grad)


def zeros_like(array_like, *, requires_grad=False):
    if isinstance(array_like, GraphNode):
        array_like = array_like.tensor
    zeros = np.zeros_like(array_like)
    return GraphNode(zeros, requires_grad=requires_grad)


def empty(shape, *, requires_grad=False):
    empty = np.empty(shape)
    return GraphNode(empty, requires_grad=requires_grad)


def empty_like(arry_like, *, requires_grad=False):
    if isinstance(array_like, GraphNode):
        array_like = array_like.tensor
    empty = np.empty_like(arry_like)
    return GraphNode(empty, requires_grad=requires_grad)


def full(shape, value, *, requires_grad=False):
    array = np.full(shape, value)
    return GraphNode(array, requires_grad=requires_grad)


def full_like(array_like, value, *, requires_grad=False):
    if isinstance(array_like, GraphNode):
        array_like = array_like.tensor
    array = np.full_like(array_like, value)
    return GraphNode(array, requires_grad=requires_grad)


def rand(*shape, requires_grad=False):
    array = np.random.rand(*shape)
    return GraphNode(array, requires_grad=requires_grad)


def randn(*shape, requires_grad=False):
    array = np.random.randn(*shape)
    return GraphNode(array, requires_grad=requires_grad)


