import torch
from torchvision.utils import save_image
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from loguru import logger
import os
import sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
# from models import *
# from utils import *

# 配置 loguru
@rank_zero_only
def setup_logger():
    logger.remove() 
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>") # 控制台日志
    logger.add("logs/training_{time}.log", rotation="500 MB", level="DEBUG")
setup_logger()

class LitDdpmCfg(pl.LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args
        unet_args = args.UNet
        diffusion_args = args.diffusion

        # 保存超参数
        logger.info("保存超参数...")
        self.save_hyperparameters(args)
    
        # 模型参数
        logger.info("初始化模型...")
        self.model = instantiate(unet_args)
        self.ddpm_trainer = instantiate(diffusion_args.trainer, model=self.model)
        self.ddpm_sampler = instantiate(diffusion_args.sampler, model=self.model)
        # 配置优化器和学习率调度器
        logger.info("配置优化器和学习率调度器...")
        self.configure_optimizers()

        logger.success("模型初始化完成.")

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long() + 1
        if torch.rand(1).item() < 0.1:
            # 10% 概率下不进行条件引导
            labels = torch.zeros_like(labels)
        loss = self.ddpm_trainer(images, labels).sum() / 10000.0
        # 记录到 Lightning 的日志系统（会显示在进度条）
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        # 每 100 个 batch 记录一次详细信息
        if batch_idx % 100 == 0:
            self.log_training_progress(batch_idx, loss)
        return loss

    @rank_zero_only
    def log_training_progress(self, batch_idx, loss):
        """只在主进程记录训练进度"""
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        logger.info(
            f"Epoch {self.current_epoch} | "
            f"Batch {batch_idx:4d} | "
            f"Loss: {loss.item():.4f} | "
            f"LR: {lr:.6f}"
        )

    def configure_optimizers(self):
        optimizer = instantiate(self.args.optimizer, params=self.model.parameters())
        cosine_scheduler = instantiate(self.args.scheduler.cosine_scheduler, optimizer=optimizer)
        scheduler = instantiate(self.args.scheduler.warmup_scheduler, optimizer=optimizer, after_scheduler=cosine_scheduler)
        return [optimizer], [scheduler]
    
    def on_train_epoch_end(self):
        epoch = self.current_epoch
        logger.info(f"Epoch {epoch} 结束，进行采样...")
        if self.trainer.is_global_zero:
            self.sample_and_save(epoch)
            logger.info(f"Epoch {epoch} 采样完成.")

    @torch.no_grad()
    def sample_and_save(self, epoch):
        self.model.eval()
        ckpt_save_dir = self.args.train.ckpt_save_dir
        img_save_dir = os.path.join(ckpt_save_dir, "eval_images")
        os.makedirs(img_save_dir, exist_ok=True)
        batch_size = 4
        x_t = torch.randn(batch_size, 3, 128, 128, device=self.device)
        # 0,1,2,2
        # labels = torch.tensor([0, 1, 2, 2], device=self.device).long() + 1
        labels = torch.randint(0,200,size=[4,],device=self.device).long() + 1
        samples = self.ddpm_sampler(x_t, labels).clamp(-1, 1)
        samples = samples * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        save_path = os.path.join(img_save_dir, f"epoch_{epoch+1:02d}.png")
        save_image(samples, save_path, nrow=2)
        logger.info(f"采样图像已保存至 {save_path}")
        self.model.train()  # 恢复训练模式


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(args: DictConfig) -> None:
    # 设置全局随机种子
    logger.info("🎯 设置随机种子...")
    pl.seed_everything(args.seed, workers=True)
    # 先训练
    logger.info("🔧 创建Lightning模型...")
    LitModel = LitDdpmCfg(args)
    logger.info("⚙️  配置Trainer训练器...")
    trainer = pl.Trainer(
        max_epochs=args.train.epoch,               # 70
        accelerator="gpu",                         # 使用 GPU
        devices=args.train.gpus,                   # 双卡训练（Animal Faces 数据集较小）
        precision="32",                            # 32 位精度
        gradient_clip_val=args.train.grad_clip,    # 1.0
        default_root_dir=args.train.ckpt_save_dir, # "./ckpt_aniaml"
        accumulate_grad_batches=args.train.accum_steps, # 梯度累积
        log_every_n_steps=10,                      # 每 10 步记录一次日志
        enable_checkpointing=True,                 # 启用检查点保存
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.train.ckpt_save_dir,
                filename='epoch_{epoch:02d}',
                every_n_epochs=1,                  # 每个 epoch 保存
                save_top_k=-1,                     # 保存所有检查点
            )
        ]
    )
    logger.info("📦 加载数据集...")
    dataset, dataloader = instantiate(args.dataloader)
    logger.info(f"数据集大小: {len(dataset)} 张图片")

    logger.info("🏃 开始训练...")
    trainer.fit(LitModel, train_dataloaders=dataloader)
    logger.success("🎉 训练完成！")
    # 采样

if __name__ == "__main__":
    main()