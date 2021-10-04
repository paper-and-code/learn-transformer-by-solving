import torch 

a = torch.rand(3, 65, 64) 
b = torch.rand(3, 32, 64)

# @ = matrix duplication
output = a@b.transpose(-2, -1)

# output => (3, 65, 32)
print(f'output {output.shape}')
