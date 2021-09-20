import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

dim = 256
patch_height = 32
patch_width = 32
b = 2
patch_dim = patch_height * patch_width * 3

to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
)

input = torch.rand(b, 3, 256, 256)
print(f'input : {input.shape}')
output = to_patch_embedding(input)
print(f'output after patch embedding : {output.shape}')

cls_token = nn.Parameter(torch.randn(1, 1, dim))

print(f'cls_token : {cls_token.shape} clas_token[0][0][:5] : {cls_token[0][0][:5]}')
cls_tokens = repeat(cls_token, '() n d -> b n d', b = b)
print(f'cls_tokens : {cls_tokens.shape}')

print(cls_tokens[0][0][0].detach().numpy()==cls_tokens[1][0][0].detach().numpy())
print(cls_tokens[0][0][0].detach().numpy(),cls_tokens[1][0][0].detach().numpy())
