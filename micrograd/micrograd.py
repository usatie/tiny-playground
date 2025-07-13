import math
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        out = Value(-self.data, (self), 'neg')
        def backward():
            self.grad += -other.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, int|float), "pow only accept int|float"
        x = self.data
        n = other
        out = Value(x ** n, (self, ), 'pow')
        def backward():
            # (x**n)' = n * x**(n-1)
            self.grad += n * (x ** (n-1)) * out.grad
        out._backward = backward
        return out

    def __truediv__(self, other):
        # a / b
        # a * b**-1
        return self * (other ** -1)


    # exp(x) where x = self
    def exp(self):
        x = self
        out = Value(math.exp(self.data), (self, ), 'exp')
        def backward():
            # exp(x)' = exp(x)
            self.grad += math.exp(self.data) * out.grad
        out._backward = backward
        return out

    # (exp(x) + exp(-1)) / (exp(x) - exp(-x))
    def tanh(self):
        x = self
        out = Value(
                (math.exp(self.data) - math.exp(-self.data))
                / (math.exp(self.data) + math.exp(-self.data)), (self, ), 'tanh')
        def backward():
            # dtanh(x) = 1 - tanh(x)**2
            self.grad = (1 - out.data**2) * out.grad
        out._backward = backward
        return out


    def backward(self):
        # Need to call backward method recursively, but how?
        # We first need to topologically sort the nodes, and reverse it
        order = []
        visited = set()
        def build_order(n):
            if n not in visited:
                visited.add(n)
                for p in n._prev:
                    build_order(p)
                order.append(n)
        build_order(self)

        # This node should be a scalar, of course
        # The start of the gradient should be 1
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
