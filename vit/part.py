import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

dim = 256
patch_height = 16
patch_width = 16
patch_dim = patch_height * patch_width * 3
to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
)

input = torch.rand(1, 3, 224, 224)
output = to_patch_embedding(input)
print(output.shape)
