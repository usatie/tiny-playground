import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

"""
print(f(3.0))

h = 0.0001
x = 3.0
print((f(x + h) - f(x)) / h)

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
plt.show()
"""

class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data)
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
print(a)
print(a + b)
print(a.__add__(b))
print(a * b)
print(a * b + c)
print(a.__mul__(b).__add__(c))
