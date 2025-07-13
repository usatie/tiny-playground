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
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            print(f"add.backward(), out={out}, self={self}, other={other}")
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            print(f"mul.backward(), out={out}, self={self}, other={other}")
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
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
