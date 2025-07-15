from tinygrad.helpers import Timing
from tinygrad import Tensor
from tinygrad import dtypes
from tinygrad.nn.datasets import mnist
import numpy as np

class Linear:
  def __init__(self, in_features, out_features, bias=True, initialization: str='kaiming_uniform'):
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
num_steps = 511

print("---numpy indexing---")
from extra.datasets import fetch_mnist
X_train, Y_train, X_test, Y_test = fetch_mnist()
with Timing("Time: "):
  avg_acc = 0
  for step in range(num_steps):
    # random sample a batch
    samp = np.random.randint(0, X_test.shape[0], size=(64))
    batch = Tensor(X_test[samp], requires_grad=False)
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass
    out = net(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc / num_steps}")

print("---tinygrad indexing---")
X_train, Y_train, X_test, Y_test = mnist()
with Timing("Time: "):
  avg_acc = 0
  for step in range(num_steps):
    # random sample a batch
    samp = Tensor.randint(64, high=X_test.shape[0])
    batch = X_test[samp].reshape(64, -1)  # flatten the images to (64, 784)
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass
    out = net(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1)
    avg_acc = avg_acc + (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc.item() / num_steps}")

