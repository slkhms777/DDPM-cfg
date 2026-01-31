import json
import torch
import torch.nn.functional as F
from models import *
from utils import *
import os
from torchvision.utils import save_image

def sample_once(cfg):
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
    device = train_cfg["device"]
    ckpt_save_dir = train_cfg["ckpt_save_dir"]


    # 模型
    model = UNet(T=total_steps, cond_size=cond_size, ch=channel, ch_mult=channel_mult,
                 num_res_blocks=num_res_blocks, dropout=dropout).to(device)
    # 加载模型权重，加载最新的ckpt，以字典序排列
    ckpts = os.listdir(ckpt_save_dir)
    latest_ckpt = sorted(ckpts)[-1]
    print(f"Loading checkpoint: {latest_ckpt}")

    ckpt = torch.load(f"{ckpt_save_dir}/{latest_ckpt}", map_location=device)
    model.load_state_dict(ckpt["model"])
    model 
    # 采样器
    sampler = DdpmSamplerCFG(total_step=total_steps, beta_start=beta_start, 
                             beta_end=beta_end, model=model, guide_scale=guide_scale).to(device)

    save_dir = "sampled_images"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(10):
        # 采样
        all_samples = []
        for j in range(10):
            labels = torch.arange(0, 10).long().to(device) + 1 + i * 10  # 标签从1开始
            batch_size = labels.shape[0]
            x_t = torch.randn(batch_size, 3, 64, 64).to(device)
            samples = sampler(x_t, labels)
            samples = samples.clamp(-1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]
            all_samples.append(samples.cpu())
        all_samples = torch.cat(all_samples, dim=0)  # [100, 3, 64, 64]
        save_image(all_samples, f"{save_dir}/class_{i+1}_to_{i+10}.png", nrow=10)
    

if __name__ == '__main__':
    cfg_path = "config.json"
    cfg = load_cfg(cfg_path)
    sample_once(cfg)