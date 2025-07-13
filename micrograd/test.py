from micrograd import Value

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

from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def test_simple_graph():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    print(a)
    print(a + b)
    print(a.__add__(b))
    print(a * b)
    print(a * b + c)
    print(a.__mul__(b).__add__(c))
    print(d)
    print(d._prev)
    print(d._op)

    draw_dot(d)

    L.grad = 1.0
    L.backward()
    dot = draw_dot(L)
    dot.render('out/dot')

if __name__ == '__main__':
    test_simple_graph()
