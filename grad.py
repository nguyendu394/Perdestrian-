from __future__ import print_function
import torch

x = torch.tensor([5.0],requires_grad=True)
y = 3*x*x
z = 5*x*x
# y.backward()
z.backward()
print(x.grad)
# print(y)
