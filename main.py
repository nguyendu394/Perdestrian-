from __future__ import print_function
import torch
import numpy as np
# print(torch.__version__)
# x = torch.empty(5, 3)
# x = torch.rand(5,3)
# x = torch.zeros(5,3,dtype=torch.long)
# x = torch.tensor((16,8))
# y = x.new_ones(5,3,dtype=torch.long)
# x = torch.randn_like(y,dtype=torch.float)
# print(y)
# print(type(x.size()))
#
# x = torch.ones([2,3])
# y = 5*x
# print(x)
# print(y)
# x.add_(x)
# print(x)
# print(y)
# torch.sub(x,y,out=x)

# x = torch.ones([4,4])
# y = x.view(-1,8)
# print(x)
# print(y)

# a = torch.rand([2,3])
# b = a.numpy()
# print(a)
# b = b + 5

# x = np.ones(5)
# print(x)
# y = torch.from_numpy(x)
# x = np.add(x, 2)
# print(x)
# print(y)
x = torch.rand(5,3)
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x).to(device)     # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
