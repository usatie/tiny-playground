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

