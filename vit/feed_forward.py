import torch 
import torch.nn as nn
input = torch.rand(2, 65, 256)
lin = nn.Linear(256, 512)
output = lin(input)
print(output.shape)
