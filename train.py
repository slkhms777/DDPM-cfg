import json
import torch
import torch.nn.functional as F
from models import *
from utils import *
from tqdm import tqdm
from torch import optim
import os 
from utils.Scheduler import WarmupScheduler
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

def train(cfg, dataloader):
    # 模型配置
    model_cfg = cfg["model_config"]
    train_cfg = cfg["train_config"]
    total_steps = model_cfg["total_steps"]
    cond_size = model_cfg["cond_size"]
    channel = model_cfg["channel"]
    channel_mult = model_cfg["channel_mult"]
    # attn = model_cfg["attn"]
    num_res_blocks = model_cfg["num_res_blocks"]
    dropout = model_cfg["dropout"]
    beta_start = model_cfg["beta_start"]
    beta_end = model_cfg["beta_end"]
    guide_scale = model_cfg["guide_scale"]
    # 训练配置
    lr = train_cfg["lr"]
    num_epochs = train_cfg["epoch"]
    multiplier = train_cfg["multiplier"]
    device = train_cfg["device"]
    grad_clip = train_cfg["grad_clip"]
    ckpt_save_dir = train_cfg["ckpt_save_dir"]

    os.makedirs(ckpt_save_dir, exist_ok=True)
    
    # 模型
    model = UNet(T=total_steps, cond_size=cond_size, ch=channel, ch_mult=channel_mult,
                 num_res_blocks=num_res_blocks, dropout=dropout).to(device)
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 训练器
    trainer = DdpmTrainerCFG(total_steps=total_steps, beta_start=beta_start, 
                          beta_end=beta_end, model=model).to(device)
    # 学习率调度器
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1)
    scheduler = WarmupScheduler(
        optimizer=optimizer, multiplier=multiplier, warm_epoch=num_epochs // 10, after_scheduler=cosineScheduler)

    # 采样器
    sampler = DdpmSamplerCFG(total_step=total_steps, beta_start=beta_start, 
                             beta_end=beta_end, model=model, guide_scale=guide_scale).to(device)
    
    for epoch in range(num_epochs):
        model.train()
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.long().to(device) + 1 # 标签从 1 - total_labels
                if torch.rand(1).item() < 0.1:
                    # 10% 概率下不进行条件引导
                    labels = torch.zeros_like(labels)
                # loss = trainer(x_0, labels).sum() / 5000
                loss = trainer(x_0, labels).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        scheduler.step()
        torch.save(model.state_dict(), f"{ckpt_save_dir}/epoch_{epoch}.pth")
        if (epoch + 1) % 2 == 0:
            eval(sampler, ckpt_save_dir, epoch + 1, device)

def eval(sampler, ckpt_dir, epoch, device, ckpt_path=None):
    # sampler.model.eval()
    img_save_dir = f"{ckpt_dir}/eval_images"
    os.makedirs(img_save_dir, exist_ok=True)
    batch_size = 16
    if ckpt_path is not None:
        sampler.model.load_state_dict(torch.load(ckpt_path, map_location=device))
    with torch.no_grad():
        x_t = torch.randn(batch_size, 3, 64, 64).to(device)
        labels = torch.randint(low=0, high=300, size=[batch_size,]).long().to(device) % 100 + 1
        samples = sampler(x_t, labels).clamp(-1, 1)
        samples = samples * 0.5 + 0.5 # [-1, 1] -> [0, 1]
        save_image(samples, f"{img_save_dir}/epoch_{epoch:02d}.png", nrow=4)



if __name__ == "__main__":
    cfg_path = "config.json"
    cfg = load_cfg(cfg_path)
    batch_size = cfg["train_config"]["batch_size"]
    dataloader = get_miniImageNet_dataloader(batch_size=batch_size, target_size=64)
    # test eval()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # sampler = DdpmSamplerCFG().to(device)
    # epoch = 0
    # sampler.model = UNet(T=50, cond_size=100, ch=32, ch_mult=(1, 2, 2, 2), attn=[1],num_res_blocks=2, dropout=0.1).to(device)
    # ckpt_dir = "./ckpt"
    # eval(sampler, ckpt_dir, epoch, device)
    #############
    train(cfg, dataloader)