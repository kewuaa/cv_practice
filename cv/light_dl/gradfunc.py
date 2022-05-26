from backend import backend as np
from math import log
from scipy.sparse import csc_matrix


def addfunc(last_grad, is_scalar, *, left, right):
    yield last_grad
    yield last_grad


def subfunc(last_grad, is_scalar, *, left, right):
    yield last_grad
    yield -last_grad
    
    
def negfunc(last_grad, is_scalar, *, left, right):
    yield -last_grad


def mulfunc(last_grad, is_scalar, *, left, right):
    if is_scalar:
        yield last_grad*right.tensor
        yield last_grad*left.tensor
    else:
        right, left = np.broadcast_arrays(
            right.tensor, left.tensor)
        yield np.diag(right.ravel
                      (order='F'))@last_grad
        yield np.diag(left.ravel
                      (order='F'))@last_grad


def matmulfunc(last_grad, is_scalar, *, left, right):
    if is_scalar:
        yield last_grad@right.tensor.T
        yield left.tensor.T@last_grad
    else:
        IM = np.eye(left.tensor.shape[0])
        IN = np.eye(right.tensor.shape[1])
        yield np.kron(right.tensor, IM)@last_grad
        yield np.kron(IN, left.tensor.T)@last_grad
    
    
def truedivfunc(last_grad, is_scalar, *, left, right):
    diff_left = 1/right.tensor
    diff_right = -left.tensor/(right.tensor**2)
    if is_scalar:
        yield last_grad*diff_left
        yield last_grad*diff_right
    else:
        diff_left, diff_right = np.broadcast_arrays(
            diff_left, diff_right)
        yield np.diag(diff_left.ravel
                       (order='F'))@last_grad
        yield np.diag(diff_right.ravel
                       (order='F'))@last_grad
    

def sumfunc(last_grad, is_scalar, *, left, right):
    yield np.ones_like(left.tensor)
    
    
def powfunc(last_grad, is_scalar, *, left, right):
    diffpow = right*left.tensor**(right-1)
    if is_scalar:
        yield last_grad*diffpow
    else:
        yield np.diag(diffpow.ravel
                       (order='F'))@last_grad
    
    
def expfunc(last_grad, is_scalar, *, left, right):
    diffexp = np.exp(left.tensor)
    if is_scalar:
        yield last_grad*diffexp
    else:
        yield np.diag(diffexp.ravel
                       (order='F'))@last_grad
    
    
def logfunc(last_grad, is_scalar, *, left, right):
    difflog = 1/left.tensor
    if right is not None:
        difflog /= log(right)
    if is_scalar:
        yield last_grad*difflog
    else:
        yield np.diag(difflog.ravel
                       (order='F'))@last_grad
    

def Tfunc(last_grad, is_scalar, *, left, right):
    if is_scalar:
        yield last_grad.T
    else:
        size = left.tensor.size
        row = np.arange(size)
        col = np.arange(size).reshape(
                    left.tensor.shape).T.ravel()
        KMN = csc_matrix(
            (np.ones(size), (row, col)), shape=(size, size))
        yield KMN.T@last_grad
    

def relufunc(last_grad, is_scalar, *, left, right):
    diffrelu = np.where(left.tensor > 0, 1., 0.)
    if is_scalar:
        yield last_grad*diffrelu
    else:
        yield np.diag(diffrelu.ravel
                       (order='F'))@last_grad
        
        
def softmaxfunc(last_grad, is_scalar, *, left, right):
    yield right


