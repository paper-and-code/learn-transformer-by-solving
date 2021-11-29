import torch

input = torch.randn(1, 3, 224, 224)
print(f'input : {input.shape}')

model = torch.nn.Conv2d(3, 3, kernel_size=7, padding=0, stride=7)
output = model(input)
print(f'output : {output.shape}')
print((224 - 12) / 7)
