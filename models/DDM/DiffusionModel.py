# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-09-03 15:51:59
# @Last Modified by:   yong
# @Last Modified time: 2023-03-30 15:43:37
# @Paper: Learning Quantum Distributions with Variational Diffusion Models

# -----internel library-----
import torch
import numpy as np
import torch.optim as optim
from tqdm.autonotebook import tqdm
import math
from einops import rearrange
from abc import ABC, abstractmethod
from collections import namedtuple
import time

# -----external library-----
from utils import get_default_device, make_beta_schedule, extract, normal_kl, mean_flat, discretized_gaussian_log_likelihood
from dataset import int2bin, bin2int, onehot, ati_onehot
from model import ConditionalModel, BasicModel, BasicModel2
from ema import EMA

import sys
sys.path.append('../../')

from Basis.Basic_Function import array_posibility_unique
from Basis.Basis_State import State
from evaluation.Fidelity import Fid
from datasets.data_generation import PaState


ModelValues = namedtuple(
    "ModelValues", ["mean", "var", "log_var", "pred_x_0", "model_eps"]
)


def get_eps_and_var(model_output, C):
    model_eps, model_v = rearrange(
        model_output, "B (split C) ... -> split B C ...", split=2, C=C
    )
    return model_eps, model_v


class GaussianDiffusion(ABC):  # include DDPM and DDIM
    def __init__(self, betas, self_condition=False):
        self.n_steps = len(betas)
        self.self_condition = self_condition
        self.device = betas.device

        # --basic parameter--
        self.betas = betas
        self.log_betas = torch.log(self.betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1]).float().to(self.device), self.alphas_cumprod[:-1]], 0)

        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(self.one_minus_alphas_cumprod)

        # --reverse diffusion--
        # x_t, eps -> x_0
        self.recip_sqrt_alphas_cumprod = 1.0 / self.sqrt_alphas_cumprod  # x_t
        self.sqrt_recip_alphas_cumprod_minus_one = torch.sqrt((1.0 / self.alphas_cumprod) - 1.0)  # eps

        # x_t, x_0 -> mean, variance
        self.posterior_mean_coef_x_0 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / self.one_minus_alphas_cumprod  # x_0
        self.posterior_mean_coef_x_t = torch.sqrt(self.alphas) * (1.0 - self.alphas_cumprod_prev) / self.one_minus_alphas_cumprod  # x_t
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / self.one_minus_alphas_cumprod  # variance
        #self.posterior_log_variance_clipped = torch.log(torch.cat((self.posterior_variance[1].view(1, 1), self.posterior_variance[1:].view(-1, 1)), 0)).view(-1)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-15))

        # x_t, eps -> mean
        self.posterior_coef_x_t = 1.0 / torch.sqrt(self.alphas)  # x_0
        self.posterior_coef_eps = self.betas / self.sqrt_one_minus_alphas_cumprod  # eps

    # --forward diffusion--
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        p_x_0 = extract(self.sqrt_alphas_cumprod, t, x_0) * x_0
        p_noise = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0) * noise

        return (p_x_0 + p_noise), noise

    # --reverse diffusion--
    # x_t, x_0 -> mean
    def q_posterior_mean(self, x_0, x_t, t):
        mean = extract(self.posterior_mean_coef_x_0, t, x_0) * x_0 + \
            extract(self.posterior_mean_coef_x_t, t, x_0) * x_t

        return mean

    # x_t, eps -> x_0
    def predict_x0_from_eps(self, x_t, t, eps, threshold=None):
        x_0 = extract(self.recip_sqrt_alphas_cumprod, t, x_t) * x_t - \
            extract(self.sqrt_recip_alphas_cumprod_minus_one, t, x_t) * eps
        return self.threshold(x_0, threshold)

    def threshold(self, x_t, threshold):
        if threshold is None:
            return x_t
        elif threshold == "static":
            return x_t.clamp(-1, 1)
        elif threshold == "dynamic":
            raise Exception("Not implemented")

    @abstractmethod
    def p_mean_variance(self, model, x_t, x_0_pred, t, threshold) -> ModelValues:
        """
        Get the model's predicted mean and variance for the distribution
        that predicts x_{t-1}
        """
        pass

    def vb_term(self, x_0, x_t, t, model):
        true_mean = self.q_posterior_mean(x_0, x_t, t)
        true_log_var = extract(self.posterior_log_variance_clipped, t, x_t)

        # self condition: x_0_pred
        with torch.no_grad():
            model_eps, model_v = get_eps_and_var(model(x_t, t), C=x_t.shape[1])
            x_0_pred = self.predict_x0_from_eps(x_t, t, model_eps, threshold=None)

        pred_mean, _, pred_log_var, _, _ = self.p_mean_variance(
            model=model, x_t=x_t, x_0_pred=x_0_pred, t=t, threshold=None
        )
        kl = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
        kl = mean_flat(kl) / math.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=pred_mean, log_scales=0.5 * pred_log_var
        )
        decoder_nll = mean_flat(decoder_nll) / math.log(2.0)
        vb_losses = torch.where(t == 0, decoder_nll, kl)

        # `th.where` selects from tensor 1 if cond is true and tensor 2 otherwise
        return vb_losses.mean(-1)

    def loss_mse(self, model_eps, noise):
        return mean_flat((noise - model_eps)**2).mean(-1)

    def loss_vb(self, model_output, x_0, x_t, t, vb_stop_grad):
        is_learned = isinstance(self, LearnedVarianceGaussianDiffusion)

        frozen_out = model_output
        if vb_stop_grad:
            assert is_learned, f"Cannot apply stop-gradient to fixed variance diffusion"
            model_eps, model_v = get_eps_and_var(model_output, C=x_t.shape[1])
            frozen_out = torch.cat([model_eps.detach(), model_v], dim=1)
        C = x_t.shape[1]
        assert frozen_out.shape[1] == C * 2 if is_learned else C

        vb_loss = self.vb_term(
            x_0=x_0,
            x_t=x_t,
            t=t,
            # TODO: The OpenAI people use kwargs, not sure
            # why not just directly return `frozen_out`
            model=lambda *_, r=frozen_out: r,
        )
        # > For our experiments, we set λ = 0.001 to prevent L_vlb from
        # > overwhelming L_simple
        # from [0]
        return vb_loss

    @abstractmethod
    def losses_training(self, *args, **kwargs):
        pass


class LearnedVarianceGaussianDiffusion(GaussianDiffusion):
    def model_v_to_log_variance(self, v, t):
        """
        Convert the model's v vector to an interpolated variance
        From (15 in [0])
        """

        min_log = extract(self.posterior_log_variance_clipped, t, v)
        max_log = extract(self.log_betas, t, v)

        # Model outputs between [-1, 1] for [min_var, max_var]
        frac = (v + 1) / 2
        return frac * max_log + (1 - frac) * min_log

    def p_mean_variance(self, model, x_t, x_0_pred, t, threshold):
        """
        Get the model's predicted mean and variance for the distribution
        that predicts x_{t-1}

        - Predict x_0 from epsilon
        - Use x_0 and x_t to predict the mean of q(x_{t-1}|x_t,x_0)
        - Turn the model's v vector into a variance
        """
        model_eps, model_v = get_eps_and_var(model(torch.cat((x_t, x_0_pred), dim=1), t), C=x_t.shape[1])
        pred_x_0 = self.predict_x0_from_eps(x_t=x_t, t=t, eps=model_eps, threshold=threshold)

        pred_mean = self.q_posterior_mean(x_0=pred_x_0, x_t=x_t, t=t)
        pred_log_var = self.model_v_to_log_variance(model_v, t)
        return ModelValues(
            mean=pred_mean,
            var=None,
            log_var=pred_log_var,
            # These are only for DDIM sampling, but I'm not
            # sure if you can use DDIM with learned variance
            pred_x_0=pred_x_0,
            model_eps=model_eps,
        )

    def losses_training(self, model_output, noise, x_0, x_t, t):
        model_eps, _ = get_eps_and_var(model_output, C=x_t.shape[1])
        mse_loss = self.loss_mse(model_eps=model_eps, noise=noise)
        vb_loss = self.loss_vb(
            model_output=model_output,
            x_0=x_0,
            x_t=x_t,
            t=t,
            vb_stop_grad=True,
        )
        # return {"mse": mse_loss, "vb": vb_loss}
        return mse_loss + 0.001 * vb_loss


class FixedSmallVarianceGaussianDiffusion(GaussianDiffusion):
    def p_mean_variance(self, model, x_t, x_0_pred, t, threshold):
        model_variance, model_log_variance = (
            extract(self.posterior_variance, t, x_t),
            extract(self.posterior_log_variance_clipped, t, x_t),
        )

        model_eps = model(torch.cat((x_t, x_0_pred), dim=1), t)
        pred_x_0 = self.predict_x0_from_eps(x_t=x_t, t=t, eps=model_eps, threshold=threshold)

        model_mean = self.q_posterior_mean(x_0=pred_x_0, x_t=x_t, t=t)
        return ModelValues(
            mean=model_mean,
            var=model_variance,
            log_var=model_log_variance,
            pred_x_0=pred_x_0,
            model_eps=model_eps,
        )

    def losses_training(self, model_output, noise):
        mse_loss = self.loss_mse(model_eps=model_output, noise=noise)
        # return {"mse": mse_loss}
        return mse_loss


class DiffusionTrainer():
    def __init__(self, diffusion, model, optimizer, dataset, epochs, batch_size):
        self.diffusion = diffusion
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = dataset.device
        self.ema = EMA(0.95)
        self.ema.register(self.model)

        if len(self.dataset) < self.batch_size:
            self.batch_size = len(self.dataset)

    def run(self):
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            loss = self.train()
            self.evaluation(pbar, loss)
        pbar.close()

    def train(self):
        self.model.train()
        permutation = torch.randperm(self.dataset.size()[0])
        # batch
        for i in range(0, self.dataset.size()[0], self.batch_size):
            # batch data
            indices = permutation[i:i + self.batch_size]
            x = self.dataset[indices]
            # Before the backward pass, zero all of the network gradients
            self.optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to parameters
            # compute the loss
            loss = self.loss(x)
            loss.backward()
            # Perform gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # Calling the step function to update the parameters
            self.optimizer.step()
            # Update the exponential moving average
            self.ema.update(self.model)
        # scheduler.step()
        return loss

    def loss(self, x_0):
        b_size = x_0.shape[0]
        # Select a random step for each example
        t = torch.randint(0, self.diffusion.n_steps, size=(b_size // 2 + 1,))
        t = torch.cat([t, self.diffusion.n_steps - t - 1], dim=0)[:b_size].long().to(self.device)
        # xt
        x_t, noise = self.diffusion.q_sample(x_0, t)
        # predict noise
        x_0_pred = torch.zeros_like(x_t)
        if self.diffusion.self_condition:# and torch.distributions.Uniform(0, 1).sample() > 0.2:
            with torch.no_grad():
                if isinstance(self.diffusion, FixedSmallVarianceGaussianDiffusion):
                    model_eps = self.model(torch.cat((x_t, x_0_pred), dim=1), t)
                elif isinstance(self.diffusion, LearnedVarianceGaussianDiffusion):
                    model_eps, model_v = get_eps_and_var(self.model(torch.cat((x_t, x_0_pred), dim=1), t), C=x_t.shape[1])
                x_0_pred = self.diffusion.predict_x0_from_eps(x_t, t, model_eps, threshold=None)

        model_out = self.model(torch.cat((x_t, x_0_pred), dim=1), t)

        losses = None
        if isinstance(self.diffusion, FixedSmallVarianceGaussianDiffusion):
            losses = self.diffusion.losses_training(
                model_output=model_out, noise=noise
            )
        elif isinstance(self.diffusion, LearnedVarianceGaussianDiffusion):
            losses = self.diffusion.losses_training(
                model_output=model_out,
                noise=noise,
                x_0=x_0,
                x_t=x_t,
                t=t,
            )
        else:
            raise Exception("Unsupported diffusion")

        return losses

    def evaluation(self, pbar, loss):
        self.model.eval()
        with torch.no_grad():
            pbar.set_description("loss {:.6f}".format(loss.item()))


class Sampler(ABC):
    def __init__(self, diffusion, self_condition):
        self.diffusion = diffusion
        self.self_condition = self_condition

    @abstractmethod
    def sample(self, model_out, x_t, t, t_next):
        pass

    def sample_loop_progressive(self, model, shape, threshold, device):
        cur_x = torch.randn(shape, device=device)
        x_0_pred = torch.zeros_like(cur_x)
        #sample_step = np.linspace(int(np.sqrt(self.diffusion.n_steps)), 0, int(np.sqrt(self.diffusion.n_steps)) + 1).astype(int)**2  # quad
        sample_step = np.linspace(self.diffusion.n_steps-1, 0, self.diffusion.n_steps).astype(int)
        for t, t_next in zip(sample_step[:-1], np.maximum(sample_step[1:] - 0, 0)):
        #for t in reversed(range(self.diffusion.n_steps)):
            with torch.no_grad():
                t = torch.tensor([t] * shape[0], dtype=torch.int64, device=device)
                t_next = torch.tensor([t_next] * shape[0], dtype=torch.int64, device=device)
                if not self.self_condition:
                    x_0_pred = torch.zeros_like(cur_x)

                model_out = self.diffusion.p_mean_variance(
                    model=model, x_t=cur_x, x_0_pred=x_0_pred, t=t, threshold=threshold
                )
                cur_x, x_0_pred = self.sample(model_out, cur_x, t, t_next)
                # yield cur_x

        return cur_x


class DDPMSampler(Sampler):
    def sample(self, model_out, x_t, t, t_next):
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (t.sum().item() != 0) + 0.0
        samples = model_out.mean + \
            nonzero_mask * torch.exp(0.5 * model_out.log_var) * noise

        return samples, model_out.pred_x_0


class DDIMSampler(Sampler):
    def __init__(self, diffusion, self_condition, eta=0.0):
        super().__init__(diffusion, self_condition)
        self.eta = eta

    def alphas(self, x_t, t, t_next):
        alpha_bar = extract(self.diffusion.alphas_cumprod, t, x_t)
        alpha_bar_prev = extract(self.diffusion.alphas_cumprod, t_next, x_t)
        return alpha_bar, alpha_bar_prev

    def sigma(self, alpha_bar, alpha_bar_prev):
        beta = 1.0 - alpha_bar / alpha_bar_prev
        sigma = self.eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * beta)
        return sigma

    def sample(self, model_out, x_t, t, t_next):
        alpha_bar, alpha_bar_prev = self.alphas(x_t=x_t, t=t, t_next=t_next)
        sigma = self.sigma(alpha_bar=alpha_bar, alpha_bar_prev=alpha_bar_prev)

        # (12 in [1])
        mean_pred = (
            model_out.pred_x_0 * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * model_out.model_eps
        )

        nonzero_mask = (t.sum().item() != 0) + 0.0
        samples = mean_pred + nonzero_mask * sigma * torch.randn_like(mean_pred)
        
        return samples, model_out.pred_x_0


def DM(data_train, K, n_qubit, N_samples, N_epoch, N_batch, schedule='quad', n_steps=100, self_condition=True, threshold='static'):
    device = get_default_device('win')
    print('device:', device)

    # dataset loading
    #data = onehot(data_train, K)
    data = int2bin(data_train)  # [0, 1]
    data = data * 2 - 1.0  # [-1, 1]
    np.random.shuffle(data)
    dataset = torch.Tensor(data).float().to(device)
    #dataset = dataset + 0.01 * torch.randn_like(dataset)

    # model
    betas = make_beta_schedule(schedule=schedule, n_timesteps=n_steps).to(device)
    dm = LearnedVarianceGaussianDiffusion(betas, self_condition=self_condition)
    #dm = FixedSmallVarianceGaussianDiffusion(betas, self_condition=self_condition)

    # reverse diffusion
    input_dim = dataset.shape[1] * 2  # self_condition
    if isinstance(dm, FixedSmallVarianceGaussianDiffusion):
        output_dim = dataset.shape[1]  # Learned
    elif isinstance(dm, LearnedVarianceGaussianDiffusion):
        output_dim = dataset.shape[1] * 2  # Learned
    #model = ConditionalModel(input_dim, output_dim, n_steps).to(device)
    model = BasicModel(input_dim, output_dim, 2, 128*2, device).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.Adamax(model.parameters(), lr=1e-2)
    DT = DiffusionTrainer(dm, model, optimizer, dataset, N_epoch, N_batch)
    DT.run()

    # sample
    time_b = time.perf_counter()
    sampler = DDPMSampler(dm, self_condition)
    cur_x = sampler.sample_loop_progressive(DT.model, (N_samples, n_qubit * 2), threshold, device)
    time_e = time.perf_counter()
    print('sample time', time_e - time_b)

    samples = (cur_x.cpu().numpy() > 0).astype(int)
    samples = bin2int(samples)
    #samples = ati_onehot(cur_x.cpu().numpy(), K)

    samples, P = array_posibility_unique(samples)

    return samples, P


if __name__ == "__main__":
    N_q = 4
    N_s = 1000
    state_name = 'W_P'
    rho_p = 1
    povm = 'Tetra4'
    K = 4
    N_epoch = 5000
    N_batch = 500
    N_samples = 10**5
    threshold = None

    sampler = PaState(basis=povm, n_qubits=N_q, State_name=state_name, P_state=rho_p)
    data_train, _ = sampler.samples_product(N_s, save_flag=False)

    samples, P = array_posibility_unique(data_train)
    _, rho_star = State().Get_state_rho(state_name, N_q, rho_p)
    Ficalc = Fid(basis=povm, n_qubits=N_q, rho_star=rho_star, torch_flag=0)
    cF = Ficalc.cFidelity_S_product(samples, P)
    print('cfdelity:', cF)

    samples, P = DM(data_train, K, N_q, N_samples, N_epoch, N_batch, schedule='sigmoid', n_steps=200, self_condition=False, threshold=threshold)

    cF = Ficalc.cFidelity_S_product(samples, P)
    print('cfdelity:', cF)
