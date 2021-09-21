from einops import rearrange 
import numpy as np 

images = [np.random.randn(30, 40, 3) for _ in range(32)]
print(rearrange(images, 'b h w c -> (b h) w c').shape)
