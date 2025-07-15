import numpy as np
from tinygrad.helpers import Timing
from tinygrad import Tensor

t1 = Tensor([1, 2, 3, 4, 5])
print(t1)
na = np.array([1, 2, 3, 4, 5])
t2 = Tensor(na)

full = Tensor.full((2, 3), fill_value=5)
print(full.tolist())
zeros = Tensor.zeros(3, 4)
print(zeros.tolist())
ones = Tensor.ones(4, 2)
print(ones.tolist())

full_like = Tensor.full_like(full, fill_value=2)
zeros_like = Tensor.zeros_like(full)
ones_like = Tensor.ones_like(full)
print(full_like.tolist())
print(zeros_like.tolist())
print(ones_like.tolist())

eye = Tensor.eye(3) # create a 3x3 identity matrix
print(eye.tolist())
arange = Tensor.arange(start=0, stop=10, step=1)
print(arange.tolist())

rand = Tensor.rand(2, 3)
print(rand.tolist())
randn = Tensor.randn(2, 3)
print(randn.tolist())
uniform = Tensor.uniform(2, 3, low=0, high=10)
print(uniform.tolist())

# Tensor.rand(*arg) = Tensor.uniform(*arg, low=0, high=1) ?

from tinygrad import dtypes
t3 = Tensor([1,2,3,4,5], dtype=dtypes.int32)
t4 = Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32)
t5 = (t4 + 1) * 2
print(t5.numpy())
print((t5 * t4).numpy())
t6 = (t5 * t4).relu().log_softmax()
print(t6.numpy())

class Linear:
    def __init__(self, in_features, out_features, bias=True, initialization='kaiming_uniform'):
        self.weight = getattr(Tensor, initialization)(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        return x.linear(self.weight.transpose(), self.bias)

class TinyNet:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leaky_relu()
        x = self.l2(x)
        return x

net = TinyNet()

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    # Y : (batch_size,)
    # loss_mask : (batch_size, )
    loss_mask = Y != ignore_index
    # self : (batch_size, ..., num_cls)
    num_cls = self.shape[-1]
    # y_counter : (num_cls, ) -> (1, num_cls) -> (batch_size, num_cls)
    y_counter = Tensor.arange(num_cls, dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), num_cls)
    # Y.flatten().reshape(-1, 1) : (batch_size, ) -> (1, batch_size) -> (batch_size, 1)
    # (y_counter == Y.flatten().reshape(-1, 1)) : (batch_size, num_cls)
    # loss_mask.reshape(-1, 1) : (batch_size, 1)
    y = ((y_counter == Y.flatten().reshape(-1, 1)) # (batch_size, num_cls) == (batch_size, 1)
         .where(-1.0, 0) # (batch_size, num_cls)
         * loss_mask.reshape(-1, 1) # * (batch_size, 1)
         ).reshape(*Y.shape, self.shape[-1]) # (batch_size, num_cls)
    return x.log_softmax().mul(y).sum() / loss_mask.sum()

from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters

#opt = SGD(get_parameters(net), lr=3e-4)
opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)

from tinygrad.nn.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist()
batch_size = 64

with Tensor.train():
    for step in range(1000):
        samp = Tensor.randint(batch_size, high=X_train.shape[0])
        batch = X_train[samp].reshape(batch_size, -1)
        labels = Y_train[samp]
        #samples = Tensor.randint(batch_size, high=X_train.shape[0])
        #batch, labels = X_train[samples], Y_train[samples]

        out = net(batch)

        loss = out.sparse_categorical_crossentropy(labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            pred = out.argmax(axis=-1)
            acc = (pred == labels).mean()
            print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")

