# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-09-03 15:51:59
# @Last Modified by:   yong
# @Last Modified time: 2023-03-30 15:43:10
# @Paper: Learning Quantum Distributions with Variational Diffusion Models

# -----internel library-----
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from torch.utils import data
import copy
import numpy as np
from tqdm.auto import tqdm
import time
import os

# -----external library-----
from .utils import get_default_device, normal_kl, standard_cdf, cycle, num_to_groups, mean_flat
from .dataset import int2bin, bin2int, onehot, ati_onehot
from .model import BasicModel, BasicModel_emb, BasicModel_emb2, BNet, SNRNetwork, beta_linear_log_snr, alpha_cosine_log_snr
from .ema_pytorch import EMA
from .transformer import Transformer


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        in_size,
        timesteps=1000,
        device="cuda",
    ):
        super().__init__()
        self.in_size = in_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = timesteps
        self.snrnet = SNRNetwork()

        self.to(device)
        self.device = device

    def p_zs_zt(self, zt, t, s, clip_denoised: bool):

        logsnr_t, norm_nlogsnr_t = self.snrnet(t)
        logsnr_s, norm_nlogsnr_s = self.snrnet(s)

        alpha_sq_t = torch.sigmoid(logsnr_t)[:, None]
        alpha_sq_s = torch.sigmoid(logsnr_s)[:, None]

        alpha_t = alpha_sq_t.sqrt()
        alpha_s = alpha_sq_s.sqrt()

        sigmasq_t = 1 - alpha_sq_t
        sigmasq_s = 1 - alpha_sq_s

        alpha_sq_tbars = alpha_sq_t / alpha_sq_s
        sigmasq_tbars = sigmasq_t - alpha_sq_tbars * sigmasq_s

        alpha_tbars = alpha_t / alpha_s

        e_hat = self.denoise_fn(zt, norm_nlogsnr_t)
        sigma_t = sigmasq_t.sqrt()
        if clip_denoised:
            e_hat.clamp_((zt - alpha_t) / sigma_t, (zt + alpha_t) / sigma_t)

        mu_zs_zt = (
            1 / alpha_tbars * zt - sigmasq_tbars / (alpha_tbars * sigma_t) * e_hat
        )
        sigmasq_zs_zt = sigmasq_tbars * (sigmasq_s / sigmasq_t)

        return mu_zs_zt, sigmasq_zs_zt

    @torch.no_grad()
    def p_zs_zt_sample(self, zt, t, s, clip_denoised=True):

        batch_size = len(zt)

        mu_zs_zt, var_zs_zt = self.p_zs_zt(zt=zt, t=t, s=s, clip_denoised=clip_denoised)
        noise = torch.randn_like(zt)
        # No noise when s == 0:
        nonzero_mask = (1 - (s == 0).float()).reshape(
            batch_size, *((1,) * (len(zt.shape) - 1))
        )
        return mu_zs_zt + nonzero_mask * var_zs_zt.sqrt() * noise

    @torch.no_grad()
    def sample_loop(self, shape):

        batch_size = shape[0]
        z = torch.randn(shape, device=self.device)

        timesteps = torch.linspace(0, 1, self.num_timesteps)

        for i in tqdm(
            reversed(range(1, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):

            t = torch.full((batch_size,), timesteps[i], device=z.device)
            s = torch.full((batch_size,), timesteps[i - 1], device=z.device)

            z = self.p_zs_zt_sample(z, t=t, s=s)

        '''
        logsnr_0, _ = self.snrnet(torch.zeros((batch_size,), device=z.device))
        alpha_sq_0 = torch.sigmoid(logsnr_0)[:, None]
        sigmasq_0 = 1 - alpha_sq_0
        sigma_0 = sigmasq_0.sqrt()

        # Get p(x | z_0)
        d = 1 / 1
        X = torch.linspace(-1, 1, 2)
        p_x_z0 = []
        for x in X:
            if x == -1:
                p = standard_cdf((x + d - z) / sigma_0)
            elif x == 1:
                p = 1 - standard_cdf((x - d - z) / sigma_0)
            else:
                p = standard_cdf((x + d - z) / sigma_0) - standard_cdf((x - d - z) / sigma_0)
            p_x_z0.append(p)

        p_x_z0 = torch.stack(p_x_z0, dim=1)

        # Sample
        cumsum = torch.cumsum(p_x_z0, dim=1)
        print(cumsum)
        r = torch.rand_like(cumsum)
        z = torch.max(cumsum > r, dim=1)[1]'''

        return z

    @torch.no_grad()
    def sample(self, batch_size=16):
        return self.sample_loop((batch_size, self.in_size))

    def q_zt_zs(self, zs, t, s=None):

        if s is None:
            s = torch.zeros_like(t)

        logsnr_t, norm_nlogsnr_t = self.snrnet(t)
        logsnr_s, norm_nlogsnr_s = self.snrnet(s)

        alpha_sq_t = torch.sigmoid(logsnr_t)[:, None]
        alpha_sq_s = torch.sigmoid(logsnr_s)[:, None]

        alpha_sq_tbars = alpha_sq_t / alpha_sq_s
        sigmasq_tbars = -torch.special.expm1(F.softplus(-logsnr_s) - F.softplus(-logsnr_t))[:, None]

        alpha_tbars = torch.sqrt(alpha_sq_tbars)
        sigma_tbars = torch.sqrt(sigmasq_tbars)

        return alpha_tbars * zs, sigma_tbars, norm_nlogsnr_t

    def prior_loss(self, x, batch_size):
        logsnr_1, _ = self.snrnet(torch.ones((batch_size,), device=x.device))
        alpha_sq_1 = torch.sigmoid(logsnr_1)[:, None]
        sigmasq_1 = 1 - alpha_sq_1
        alpha_1 = alpha_sq_1.sqrt()
        mu_1 = alpha_1 * x
        return normal_kl(mu_1, sigmasq_1).sum() / batch_size

    def data_likelihood(self, x, batch_size):
        logsnr_0, _ = self.snrnet(torch.zeros((1,), device=x.device))
        alpha_sq_0 = torch.sigmoid(logsnr_0)[:, None].repeat(*x.shape)
        sigmasq_0 = 1 - alpha_sq_0
        alpha_0 = alpha_sq_0.sqrt()
        mu_0 = alpha_0 * x
        sigma_0 = sigmasq_0.sqrt()
        d = 1 / 1
        p_x_z0 = standard_cdf((x + d - mu_0) / sigma_0) - standard_cdf((x - d - mu_0) / sigma_0)
        p_x_z0[x == 1] = 1 - standard_cdf((x[x == 1] - d - mu_0[x == 1]) / sigma_0[x == 1])
        p_x_z0[x == -1] = standard_cdf((x[x == -1] + d - mu_0[x == -1]) / sigma_0[x == -1])
        nll = -torch.log(p_x_z0)
        return nll.sum() / batch_size

    def sample_t(self, batch_size):
        t1 = torch.ones(batch_size, device=self.device) * torch.rand((1,), device=self.device)
        t2 = torch.linspace(0, (batch_size - 1) / batch_size, batch_size).to(self.device)
        return torch.fmod(t1 + t2, 1)

    def get_loss(self, x):

        batch_size = len(x)

        e = torch.randn_like(x)
        t = self.sample_t(batch_size)
        #t = torch.rand((batch_size,), device=self.device)
        '''
        b_size = batch_size
        # Select a random step for each example
        t = torch.randint(0, 1000, size=(b_size // 2 + 1,))
        t = torch.cat([t, 1000 - t - 1], dim=0)[:b_size].long().to(self.device)
        t = t / 1000'''

        mu_zt_zs, sigma_zt_zs, norm_nlogsnr_t = self.q_zt_zs(zs=x, t=t)

        zt = mu_zt_zs + sigma_zt_zs * e

        e_hat = self.denoise_fn(zt.detach(), norm_nlogsnr_t)

        t.requires_grad_(True)
        logsnr_t, _ = self.snrnet(t)
        logsnr_t_grad = autograd.grad(logsnr_t.sum(), t)[0]

        diffusion_loss = (
            -0.5
            * logsnr_t_grad
            * mean_flat(F.mse_loss(e, e_hat, reduction="none"))
        )

        '''
        logsnr_t, _ = self.snrnet(t)
        loss_weight = (1.0 + logsnr_t.exp()) ** (-1)
        diffusion_loss = loss_weight * mean_flat(F.mse_loss(e, e_hat, reduction="none"))'''
        
        diffusion_loss = diffusion_loss.mean()
        #prior_loss = self.prior_loss(x, batch_size)
        #data_loss = self.data_likelihood(x, batch_size)

        #loss = diffusion_loss + prior_loss + data_loss

        return diffusion_loss

    def forward(self, x):
        return self.get_loss(x)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        ema_decay=0.999,
        train_batch_size=32,
        train_lr=1e-4,
        train_total_steps=2**14,
        gradient_accumulate_every=4,
        step_start_ema=1024,
        update_ema_every=16,
        save_n_images=1000,
        save_n_images_every=100,
        device="cuda",
    ):
        super().__init__()
        self.model = diffusion_model
        self.device = device

        self.ema = EMA(self.model, beta=ema_decay, update_after_step=step_start_ema, update_every=update_ema_every)

        self.batch_size = train_batch_size
        self.in_size = diffusion_model.in_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_total_steps = train_total_steps
        self.save_n_images = save_n_images
        self.save_n_images_every = save_n_images_every

        # transform
        self.dataset = torch.Tensor(dataset).float().to(device)
        self.dataloader = cycle(
            data.DataLoader(
                self.dataset, batch_size=train_batch_size, shuffle=True
            )
        )
        self.optimizer = optim.NAdam(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=train_total_steps, eta_min=0.)

    def sample_all(self):
        batches = num_to_groups(self.save_n_images, self.save_n_images_every)
        all_images_list = list(
            map(lambda n: self.model.sample(batch_size=n), batches)
        )
        all_images = torch.cat(all_images_list, dim=0)

        return all_images

    def train(self):
        pbar = tqdm(range(self.train_total_steps))
        for step in pbar:
            self.optimizer.zero_grad()
            #for i in range(self.gradient_accumulate_every):
            data = next(self.dataloader)
            data = data.to(self.device)
            loss = self.model(data)
            #(loss / self.gradient_accumulate_every).backward()
            loss.backward()

            self.optimizer.step()
            self.ema.update()
            self.scheduler.step()

            if step % 10 == 0:
                pbar.set_description("loss {:.6f}".format(loss.item()))

        pbar.close()
        self.model = self.ema.ema_model

        all_images = self.sample_all()
        return all_images


def VDM(data_train, N_samples, N_epoch, N_batch, n_steps=100):
    # device
    device = get_default_device('win')
    print('device:', device)

    # dataset loading
    #data = onehot(data_train, K)
    data_train = int2bin(data_train)  # [0, 1]
    dataset = data_train * 2 - 1.0  # [-1, 1]

    # build model
    input_dim = dataset.shape[1]
    output_dim = dataset.shape[1]
    #net = BasicModel_emb(input_dim, output_dim, n_steps).to(device)
    #net = BasicModel_emb2(input_dim, output_dim, 4, 128, device)
    #net = BNet(input_dim, output_dim, 4, input_dim, device)

    net = Transformer(input_dim, output_dim, num_layers=2, device=device)

    diffusion = GaussianDiffusion(
        net,
        in_size=input_dim,
        timesteps=n_steps,
        device=device,
    )

    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size=N_batch,
        train_lr=1e-3,
        train_total_steps=int(len(data_train)/N_batch)*N_epoch,
        gradient_accumulate_every=2,
        ema_decay=0.999,
        save_n_images=N_samples,
        save_n_images_every=2*10**5,
        device=device,
    )

    cur_x = trainer.train()

    #samples = cur_x.cpu().numpy()
    samples = (cur_x.cpu().numpy() > 0).astype(int)
    samples = bin2int(samples)
    #samples = ati_onehot(cur_x.cpu().numpy(), K)

    #samples, P = array_posibility_unique(samples)

    return samples
