import numpy as np
class value:
    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None
        self.grad = 0
    
    def __repr__(self):
        return f"value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other):
        out = value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = value(1/(1+np.exp(-self.data)), (self,), 'sigmoid')
        def _backward():
            self.grad += (out.data)*(1-out.data)*out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

        