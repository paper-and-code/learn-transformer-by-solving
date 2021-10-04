import torch.nn as nn

input=  torch.random(1,3,224,224)
identity_layer = nn.Identity()
output = identity_layer(input)
print(f'output => {output.shape}')
