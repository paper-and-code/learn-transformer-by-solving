import torch

a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
# print(F'a.shape : {a.shape}') # [3,2]
b = a.flatten(0)
print(F'b : {b}')

# A, ,,,,,
# B, ,,,
# torch.stack((a, b), -1).flatten(-2)
