import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def gather(k, t, x_shape):
    if k.device != t.device:
        k = k.to(t.device)
    out = torch.gather(k, index=t, dim=0).float().to(t.device).reshape(
        [x_shape[0]] + [1] * (len(x_shape) - 1))
    return out

class DdpmTrainerCFG(nn.Module):
    def __init__(self, total_steps=50, beta_start=1e-4, beta_end=0.02, model=None):
        # 默认beta设置与DDPM论文一致
        super().__init__()
        self.total_steps = total_steps
        self.model = model
        self.betas = torch.linspace(beta_start, beta_end, self.total_steps).float()
        self.alphas = 1.0 - self.betas.float()
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).float()
        self.one_minus_alpha_bar = 1.0 - self.alpha_bar.float()
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(self.alpha_bar).float())
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(self.one_minus_alpha_bar).float())

    def forward(self, x_0, condition):
        # 输入原始图像，返回加噪t步的噪声图像和噪声
        # x : [batch, channels, height, width]
        # t : [batch]
        assert x_0.shape[0] == condition.shape[0]
        z = torch.randn_like(x_0)
        t = torch.randint(self.total_steps, (x_0.shape[0],), device=x_0.device) # 0-999

        # x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * z
        x_t = gather(self.sqrt_alpha_bar, t, x_0.shape) * x_0 \
            + gather(self.sqrt_one_minus_alpha_bar, t, x_0.shape) * z
        loss = F.mse_loss(self.model(x_t, t, condition), z, reduction='none')
        # loss.shape = [batch, channels, height, width]
        return loss

class DdpmSamplerCFG(nn.Module):
    def __init__(self, total_step=50, beta_start=1e-4, beta_end=0.02, model=None, guide_scale=3.0):
        super().__init__()
        self.total_steps = total_step
        self.model = model
        self.guide_scale = guide_scale
        self.betas = torch.linspace(beta_start, beta_end, self.total_steps).float()
        self.alphas = 1.0 - self.betas.float()
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).float()
        self.one_minus_alpha_bar = 1.0 - self.alpha_bar.float()
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).float()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(self.one_minus_alpha_bar).float()

        
        self.register_buffer('alphas_rsqrt', torch.rsqrt(self.alphas))
        self.register_buffer('betas_over_sqrt_one_minus_alpha_bar', self.betas / self.sqrt_one_minus_alpha_bar)    
        self.register_buffer('sigma_2', self.betas * \
                             F.pad(self.one_minus_alpha_bar[:-1], (1,0), value=0) / self.one_minus_alpha_bar)

    def get_mu_sigma(self, x_t, t, pred_noise):
        mu = gather(self.alphas_rsqrt, t, x_t.shape) \
            * (x_t - gather(self.betas_over_sqrt_one_minus_alpha_bar, t, x_t.shape) * pred_noise)
        sigma_2 = gather(self.sigma_2, t, x_t.shape)
        return mu, torch.sqrt(sigma_2)
    
    def step_once(self, x_t, t, pred_noise):
        mu, sigma = self.get_mu_sigma(x_t, t, pred_noise)
        z = torch.randn_like(x_t)
        x_prev = mu + sigma * z
        return x_prev
    
    def forward(self, x_t, condition):
        # 逆向采样过程
        assert x_t.shape[0] == condition.shape[0]
        non_condition = torch.zeros_like(condition).to(condition.device)

        x_prev = x_t
        for cur_step in tqdm(reversed(range(self.total_steps)), desc="采样进度", total=self.total_steps,#999-0
                             bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%'): 
            
            t = x_t.new_ones([x_t.shape[0],], dtype=torch.long) * cur_step

            # classifier-free guidance
            pred_noise_nc = self.model(x_prev, t, non_condition) 
            pred_noise_c = self.model(x_prev, t, condition)
            pred_noise = pred_noise_nc + self.guide_scale * (pred_noise_c - pred_noise_nc)

            x_prev = self.step_once(x_prev, t, pred_noise)
        x_0 = x_prev
        return x_0
        
if __name__ == "__main__":
    from UNet import UNet
    # mps 
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = DdpmTrainerCFG()
    model = UNet(
        T=1000, cond_size=10, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1).to(device)
    trainer.model = model
    trainer.model.train()
    batch_size = 8
    x_0 = torch.randn(batch_size, 3, 32, 32).to(device)
    condition = torch.randint(10, size=[batch_size]).to(device) + 1
    loss = trainer(x_0, condition)
    print(loss.shape, loss)

    sampler = DdpmSamplerCFG()
    sampler.model = model
    sampler.model.eval()
    x_t = torch.randn(batch_size, 3, 32, 32).to(device)
    with torch.no_grad():
        x_0 = sampler(x_t, condition)
    print(x_0.shape)