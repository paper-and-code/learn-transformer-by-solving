import torch

a = torch.randn(3, 224, 224)
b = a.to(torch.bool)
c = a.to(torch.float64)
print(b.dtype, a.dtype, c.dtype)