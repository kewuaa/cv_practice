from backend import backend as np
from gradfunc import *


class MyErr(Exception): pass
class DimErr(MyErr): pass
class NotRequireErr(MyErr): pass


class GraphNode:
    
    def __init__(self, tensor, *, requires_grad=False):
        self.tensor = np.asarray(tensor)
        self.requires_grad = requires_grad
        self._grad = 0.
        self.save_grad = False
        self.as_unique = False
        self.gradfunc = None
        self.is_leaf = True
        self.father = None
        self.subnode_l = None
        self.subnode_r = None
        self.operation = 'None'
        
    def __str__(self):
        return str(self.tensor)
    
    def __repr__(self):
        return '\n'.join((repr(self.tensor),
                          'requires grad:',
                          str(self.requires_grad),
                          'grad:',
                          str(self._grad),
                          'is leaf:',
                          str(self.is_leaf),
                          'operation:',
                          self.operation,
                          ))
    
    def __call__(self, i, j=None):
        if j is None:
            return self.tensor[i]
        else: return self.tensor[i, j]
    
    def __pos__(self):
        return self
        
    def __add__(self, other):
        if not isinstance(other, GraphNode):
            other = GraphNode(other)
        new_tensor = self.tensor+other.tensor
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad or \
                                other.requires_grad
        new_node.gradfunc = addfunc
        new_node.is_leaf = False
        new_node.subnode_l = self
        new_node.subnode_r = other
        self.father = new_node
        other.father = new_node
        new_node.operation = 'add'
        return new_node
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        if isinstance(other, GraphNode):
            tensor = other.tensor
        else:
            tensor = np.asarray(other)
        self.tensor += tensor
        return self
    
    def __sub__(self, other):
        if not isinstance(other, GraphNode):
            other = GraphNode(other)
        new_tensor = self.tensor-other.tensor
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad or \
                                other.requires_grad
        new_node.gradfunc = subfunc
        new_node.is_leaf = False
        new_node.subnode_l = self
        new_node.subnode_r = other
        self.father = new_node
        other.father = new_node
        new_node.operation = 'sub'
        return new_node
    
    def __neg__(self):
        new_node = GraphNode(-self.tensor)
        new_node.requires_grad = self.requires_grad
        new_node.gradfunc = negfunc
        new_node.is_leaf = False
        new_node.subnode_l = self
        self.father = new_node
        new_node.operation = 'neg'
        return new_node
    
    def __rsub__(self, other):
        if not isinstance(other, GraphNode):
            other = GraphNode(other)
        new_tensor = other.tensor-self.tensor
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad or \
                                other.requires_grad
        new_node.gradfunc = subfunc
        new_node.is_leaf = False
        new_node.subnode_l = other
        new_node.subnode_r = self
        self.father = new_node
        other.father = new_node
        new_node.operation = 'rsub'
        return new_node
    
    def __isub__(self, other):
        if isinstance(other, GraphNode):
            tensor = other.tensor
        else:
            tensor = np.asarray(other)
        self.tensor -= tensor
        return self
    
    def __mul__(self, other):
        if not isinstance(other, GraphNode):
            other = GraphNode(other)
        new_tensor = self.tensor*other.tensor
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad or \
                                other.requires_grad
        new_node.gradfunc = mulfunc
        new_node.is_leaf = False
        new_node.subnode_l = self
        new_node.subnode_r = other
        self.father = new_node
        other.father = new_node
        new_node.operation = 'mul'
        return new_node
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def matmul(self, other, *, order='pos'):
        if not isinstance(other, GraphNode):
            other = GraphNode(other)
        if self.tensor.ndim < 2 or other.tensor.ndim < 2:
            raise DimErr('not enough dims to matmul')
        try:
            if order == 'neg': raise ValueError
            new_tensor = self.tensor@other.tensor
        except ValueError as e:
            if order == 'pos': raise e
            new_tensor = other.tensor@self.tensor
            left = other
            right = self
        else:
            left = self
            right = other
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad or \
                                other.requires_grad
        new_node.gradfunc = matmulfunc
        new_node.is_leaf = False
        new_node.subnode_l = left
        new_node.subnode_r = right
        self.father = new_node
        other.father = new_node
        new_node.operation = 'matmul'
        return new_node
    
    def __truediv__(self, other):
        if not isinstance(other, GraphNode):
            other = GraphNode(other)
        new_tensor = self.tensor/other.tensor
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad or \
                                other.requires_grad
        new_node.gradfunc = truedivfunc
        new_node.is_leaf = False
        new_node.subnode_l = self
        new_node.subnode_r = other
        self.father = new_node
        other.father = new_node
        new_node.operation = 'truediv'
        return new_node
    
    def __rtruediv__(self, other):
        if not isinstance(other, GraphNode):
            other = GraphNode(other)
        new_tensor = other.tensor/self.tensor
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad or \
                                other.requires_grad
        new_node.gradfunc = truedivfunc
        new_node.is_leaf = False
        new_node.subnode_l = other
        new_node.subnode_r = self
        self.father = new_node
        other.father = new_node
        new_node.operation = 'rtruediv'
        return new_node
        
    
    def __pow__(self, other):
        new_tensor = self.tensor**other
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad
        new_node.gradfunc = powfunc
        new_node.is_leaf = False
        new_node.subnode_l = self
        new_node.subnode_r = other
        self.father = new_node
        new_node.operation = 'pow'
        return new_node
    
    def sum(self, axis=None):
        if axis is not None:
            ones = np.ones(self.tensor.shape[axis])
            ones = np.expand_dims(ones, axis=axis)
            return self.matmul(ones, order=None)
        else:
            new_tensor = self.tensor.sum()
            new_node = GraphNode(new_tensor)
            new_node.requires_grad = self.requires_grad
            new_node.gradfunc = sumfunc
            new_node.is_leaf = False
            new_node.subnode_l = self
            self.father = new_node
            new_node.operation = 'sum'
            return new_node
        
    @property
    def T(self):
        new_tensor = self.tensor.T.copy()
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad
        new_node.gradfunc = Tfunc
        new_node.is_leaf = False
        new_node.subnode_l = self
        self.father = new_node
        new_node.operation = 'T'
        return new_node
    
    def relu(self):
        new_tensor = np.where(self.tensor > 0, self.tensor, 0)
        new_node = GraphNode(new_tensor)
        new_node.requires_grad = self.requires_grad
        new_node.gradfunc = relufunc
        new_node.is_leaf = False
        new_node.subnode_l = self
        self.father = new_node
        new_node.operation = 'relu'
        return new_node
    
    def backward(self, init_grad=None, is_scalar=True):
        if self.is_leaf or self.as_unique:
            self._grad += init_grad
        else:
            if init_grad is None:
                if (size := self.tensor.size) > 1:
                    is_scalar = False
                    init_grad = np.eye(size)
                else:
                    init_grad = np.ones_like(self.tensor)
            grads = self.gradfunc(init_grad, is_scalar,
                left=self.subnode_l, right=self.subnode_r)
            left_grad = next(grads)
            if self.subnode_l is not None:
                if self.subnode_l.requires_grad:
                    self.subnode_l.backward(
                        init_grad=left_grad, is_scalar=is_scalar)
            if isinstance(self.subnode_r, GraphNode):
                if self.subnode_r.requires_grad:
                    self.subnode_r.backward(
                        init_grad=next(grads), is_scalar=is_scalar)
            del grads
            
    @property
    def grad(self):
        if not self.requires_grad:
            raise NotRequireErr('without require')
        grad = self._grad
        if not self.save_grad:
            self.reset_grad()
        return grad
    
    def detach(self):
        self.as_unique = not self.as_unique
    
    def reset_grad(self):
        self._grad = 0.
        if self.father is not None:
            self.father.reset_grad()
            
    @property
    def shape(self):
        return self.tensor.shape
        
    def reshape(self, *size ,shape=()):
        if shape:
            new_tensor = self.tensor.reshape(shape)
        else:
            new_tensor = self.tensor.reshape(size)
        return GraphNode(
                    new_tensor, requires_grad=self.requires_grad)
        
        
        
