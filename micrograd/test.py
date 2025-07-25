from micrograd import Value
from nn import Neuron, Layer, MLP

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

    L.grad = 1.0
    L.backward()
    dot = draw_dot(L)
    dot.render('out/dot')

def test_perceptron():
    x1 = Value(2.0, label='x1')
    x2 = Value(0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')
    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    sigma = x1w1x2w2 + b; sigma.label = 'Σ'
    o = sigma.tanh(); o.label = 'o'
    o.backward()
    dot = draw_dot(o)
    dot.render('out/perceptron')

def test_perceptron2():
    x1 = Value(2.0, label='x1')
    x2 = Value(0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')
    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    sigma = x1w1x2w2 + b; sigma.label = 'Σ'
    e = (2 * sigma).exp(); e.label = 'e'
    o = (e - 1) / (e + 1); o.label = 'o'
    o.backward()
    dot = draw_dot(o)
    dot.render('out/perceptron2')

def test_neuron():
    x = [2.0, 3.0, 4.0]
    n = Neuron(3)
    o = n(x)
    dot = draw_dot(o)
    dot.render('out/neuron')

def test_layer():
    x = [2.0, 3.0, 4.0]
    l = Layer(3, 4)
    o = l(x)
    print(o)

def test_mlp():
    x = [2.0, 3.0, 4.0]
    n = MLP(3, [4,4,1])
    o = n(x)
    print(o)
    dot = draw_dot(o)
    dot.render('out/MLP')

def test_training():
    model = MLP(3, [4,4,1])
    X_train = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    y_train = [1.0, -1.0, -1.0, 1.0]
    for _ in range(100):
        ypred = [model(x) for x in X_train]
        loss = sum([(yhat-y)**2 for y, yhat in zip(y_train, ypred)])
        loss.backward()
        for p in model.parameters():
            p.data -= 0.01 * p.grad
        model.zero_grad()
    print(ypred)

if __name__ == '__main__':
    test_simple_graph()
    test_perceptron()
    test_perceptron2()
    test_neuron()
    test_layer()
    test_mlp()
    test_training()
