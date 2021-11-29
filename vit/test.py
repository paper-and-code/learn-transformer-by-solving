import torch
from vit import ViT

model = ViT(image_size=256,
            patch_size=32,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1)
batch_num = 1
img = torch.randn(batch_num, 3, 256, 256)

preds = model(img)  # (1, 1000)
