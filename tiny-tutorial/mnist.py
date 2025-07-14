from tinygrad import Device

# default device
print(Device.DEFAULT)

from tinygrad import Tensor, nn

class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        # (bs, 28, 28, 1) -> (bs, 26, 26, 1)-> (bs, 13, 13, 32)
        x = self.l1(x).relu().max_pool2d((2, 2))
        # (bs, 13, 13, 32) -> (bs, 11, 11, 64) -> (bs, 5, 5, 64)
        x = self.l2(x).relu().max_pool2d((2, 2))
        # (bs, 5, 5, 64) -> (bs, 1600) -> (bs, 10)
        return self.l3(x.flatten(1).dropout(0.5))


from tinygrad.nn.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist()
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)

model = Model()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
print(acc.item())
# 0.11169999837875366

optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128
def step():
    Tensor.training = True # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss

import timeit
print(timeit.repeat(step, repeat=5, number=1))
"""
[0.73853929201141, 0.08297212515026331, 0.056711249984800816, 0.056308834347873926, 0.056178207974880934]
"""

from tinygrad import GlobalCounters, Context
GlobalCounters.reset()
with Context(DEBUG=2):
    step()

from tinygrad import TinyJit
jit_step = TinyJit(step)
print(timeit.repeat(step, repeat=5, number=1))
"""
[0.06480475002899766, 0.05591683275997639, 0.0562029997818172, 0.05640150001272559, 0.05721954209730029]
"""

# Q. why is the time not improved even after JIT? Is it implicitly JITed by the first timeit?

# Putting it together
for step in range(7000):
    loss = jit_step()
    if step % 100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100:.2f}%")
