from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2
import json
import torch
from PIL import Image
import io
import pyarrow.parquet as pq

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

def get_AnimalFaces_dataloader(batch_size, target_size=128, num_workers=4, root="datasets/animal_faces"):
    transform = v2.Compose([
        v2.Resize(size=None, max_size=target_size),
        v2.CenterCrop(size=target_size),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True), # scale=True -> [0, 1]
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=f"{root}/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_dataset, train_loader


class TinyImageNetParquetDataset(Dataset):
    """
    专为 TinyImageNet Parquet 格式优化的 Dataset
    - image: dict {'bytes': bytes, 'path': None}
    - label: int (0-199)
    """
    def __init__(self, parquet_path, transform=None):
        print(f"Loading {parquet_path}...")
        self.table = pq.read_table(parquet_path)
        self.df = self.table.to_pandas()
        self.transform = transform
        
        # 预提取所有标签，加速训练
        self.labels = self.df['label'].values.astype('int64')
        
        print(f"Loaded {len(self.df)} samples")
        print(f"Number of classes: {len(set(self.labels))} (should be 200)")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 快速获取图像字节（避免 pandas 开销）
        image_dict = self.df['image'].iloc[idx]
        image_bytes = image_dict['bytes']  # 直接访问 dict
        
        # 解码 JPEG
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_TinyImageNet_dataloader(
    parquet_path="datasets/TinyImageNet/train.parquet",
    batch_size=64,
    target_size=64,  # TinyImageNet 原始分辨率
    num_workers=4,
    drop_last=True,
    use_augmentation=False  # DDPM 通常不需要强增强
):
    """
    TinyImageNet Parquet DataLoader
    
    Args:
        use_augmentation: DDPM 训练建议 False，保持数据纯净
    """
    # 基础变换
    transforms = [
        v2.Resize(size=None, max_size=target_size),
        v2.CenterCrop(size=target_size),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),  # [0, 1]
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ]
    
    # DDPM 通常不需要数据增强，但如果需要可以开启
    if use_augmentation:
        transforms.insert(0, v2.RandomHorizontalFlip(p=0.5))
    
    transform = v2.Compose(transforms)
    
    dataset = TinyImageNetParquetDataset(parquet_path, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    
    return dataset, loader


def load_cfg(cfg_path):
    with open(cfg_path, "r", encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg

if __name__ == '__main__':
    # 测试加载
    dataset, loader = get_TinyImageNet_dataloader(
        parquet_path="datasets/TinyImageNet/train.parquet",
        batch_size=4,
        target_size=128,
        num_workers=0  # 调试时设为 0
    )
    
    # 检查一批数据
    images, labels = next(iter(loader))
    print(f"\nBatch shape: {images.shape}")  # [B, 3, 64, 64]
    print(f"Labels: {labels}")
    print(f"Value range: [{images.min():.2f}, {images.max():.2f}]")  # 应接近 [-1, 1]
    
    # 保存一组图片测试
    from torchvision.utils import save_image
    save_image(images, "test_images.png", nrow=2)

    # 统计类别分布
    print("\nClass distribution check:")
    all_labels = [labels.numpy() for _, labels in loader]
    all_labels = torch.cat([torch.from_numpy(label) for label in all_labels])
    print(f"Unique classes in first epoch: {len(set(all_labels.tolist()))}/200")