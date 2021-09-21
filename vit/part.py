import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

image_height, image_width = 256, 256
dim = 256
patch_height = 32
patch_width = 32
num_patches = int(image_height / patch_height * image_width / patch_width)
b = 2
patch_dim = patch_height * patch_width * 3

to_patch_embedding = nn.Sequential(
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
              p1=patch_height,
              p2=patch_width),
    nn.Linear(patch_dim, dim),
)

input = torch.rand(b, 3, image_height, image_width)
print(f'input : {input.shape}')
x = to_patch_embedding(input)
print(f'  koutput after patch embedding : {x.shape}')

cls_token = nn.Parameter(torch.randn(1, 1, dim))

print(
    f'  cls_token : {cls_token.shape} clas_token[0][0][:2] : {cls_token[0][0][:2].detach().numpy()}'
)
cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)
print(f'  cls_tokens : {cls_tokens.shape}')

assert cls_tokens[0][0][0].detach().numpy() == cls_tokens[1][0][0].detach(
).numpy()
b, n, _ = x.shape
x = torch.cat((cls_tokens, x), dim=1)
print(f'  concat => {x.shape}')

pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

print(f'  pos_embedding => {pos_embedding.shape}')

# x += pos_embedding[:, :(n + 1)]
x += pos_embedding
print(f'output after position embedding => {x.shape}')

# Transformer Part
layers = nn.ModuleList([])
# for attn, ff in layers:
#     x = attn(x) + x
#     x = ff(x) + x

# Transformer Part
## Multi Head Self Attention
dim_head = 64
heads = 16
inner_dim = dim_head * heads
to_qkv = nn.Linear(
    dim, inner_dim * 3,
    bias=False)  ## inner_dim = 1024 , output B, Patch_num, 1024*3
qkv = to_qkv(x).chunk(3, dim=-1)
print(f'  qkv => {type(qkv)} {qkv[0].shape} {qkv[0].shape[-1]==inner_dim}')
## b, n, (h,d ) =2, 65, 1024 => 2, 16(h), 65, 64
# h, d = 1024 First 64 correspond head 0, second 64 = head 1
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), qkv)
print(f'  q => {q.shape}')
# Query length per 1 patch = dim_head(64)
assert q.shape == (2, 16, 65, 64)
assert k.shape == (2, 16, 65, 64)
assert v.shape == (2, 16, 65, 64)

# scale = 1/sqrt(d)
dots = torch.matmul(q, k.transpose(-1, -2)) * scale
## Q, K, V =
# Q x K = Calculate Which patch is similar or not
# Self Attention = softmax(q * k) * v
# Q = Search Term
# K = Saved Tag
# V = Saved Content
