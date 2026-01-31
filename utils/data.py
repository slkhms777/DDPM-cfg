from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2
import json
import torch

# def get_miniImgaeNet_dataloader(batch_size, target_size=128, num_workers=4, root="datasets/mini_imagenet"):
#     # 定义数据变换（因为是生产模型不作增强）
#     transform = v2.Compose([
#         v2.Resize(size=None, max_size=target_size),
#         v2.CenterCrop(size=target_size),
#         v2.ToImage(),
#         v2.ToDtype(dtype=torch.float32, scale=True), # scale=True -> [0, 1]
#         v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
#     ])
#     train_dataset = torchvision.datasets.ImageFolder(root=f"{root}/train", transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     return train_loader
def get_miniImageNet_dataloader(batch_size, target_size=64, num_workers=4, root="datasets/mini_imagenet"):
    """
    处理方式：短边resize到target_size，然后从中心裁剪正方形
    这样保证：(1) 无黑边 (2) 图像内容完整 (3) 长宽比保持一致（通过裁剪）
    """
    transform = v2.Compose([
        # 短边缩放到 target_size，长边等比例缩放
        v2.Resize(size=target_size),  # 当传入int时，Resize会将短边缩放到这个值
        # 从中心裁剪出 target_size x target_size
        v2.CenterCrop(size=target_size),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),  # [0, 1]
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=f"{root}/train", 
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader

def load_cfg(cfg_path):
    with open(cfg_path, "r", encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg