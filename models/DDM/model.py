# An Implementation of Diffusion Network Model
# Oringinal source: https://github.com/acids-ircam/diffusion_models

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        freqs = x[:, None] * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=1)
        fouriered = torch.cat((x[:, None], fouriered), dim=1)
        return fouriered


class FourierPosEmb(nn.Module):
    def __init__(self, num) -> None:
        super().__init__()
        self.num = num

    def forward(self, x):
        embed_ins = x
        for n in self.num:
            Fourier_sin = torch.sin(2**n * torch.pi * x)
            Fourier_cos = torch.cos(2**n * torch.pi * x)
            embed_ins = torch.cat((embed_ins, Fourier_sin, Fourier_cos), dim=1)

        return embed_ins


class PositionEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device

        position = torch.arange(self.dim, device=device)
        div_term = torch.exp(torch.arange(self.dim, device=device) *
                             -(math.log(10000.0) / self.dim))
        emb_sin = torch.sin(position * div_term)
        emb_cos = torch.cos(position * div_term)
        return torch.cat((x + emb_sin, x + emb_cos), dim=1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim))
        self.b = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim_out), LayerNorm(dim_out)
        )
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.block(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out * 2), nn.Dropout(0.1))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        if self.mlp is not None:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()

        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_steps):
        super(ConditionalModel, self).__init__()

        self.lin1 = ConditionalLinear(input_dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, output_dim)

    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)


class BasicModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, channels=128, device='cpu'):
        super(BasicModel, self).__init__()

        self.timestep_coeff = torch.linspace(0.1, 100, channels).to(device)
        self.timestep_phase = nn.Parameter(torch.randn(channels))
        self.input_embed = nn.Linear(input_dim, channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Softplus(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.Softplus(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.Softplus())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, output_dim),
        )

        self.to(device)

    def forward(self, x, y):
        y = y.view(-1, 1)
        embed_t = torch.sin(
            (self.timestep_coeff * y.float()) + self.timestep_phase
        )
        embed_t = self.timestep_embed(embed_t)
        embed_ins = self.input_embed(x)
        out = self.layers(embed_ins + embed_t)
        return out


class BasicModel_emb(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, channels=128, device='cpu'):
        super(BasicModel_emb, self).__init__()

        self.channels = channels
        self.time_fc0 = nn.Sequential(SinusoidalPosEmb(self.channels), nn.Linear(channels, channels))
        self.time_fc1 = nn.Linear(channels, 2 * channels)
        self.time_fc2 = nn.Linear(2 * channels, 4 * channels)
        self.time_fc3 = nn.Linear(4 * channels, 2 * channels)
        self.act = nn.SiLU()

        '''
        self.Fourier_num = [7, 8]
        self.Fourier_n = len(self.Fourier_num)
        embed_dim = (2 * self.Fourier_n + 1) * input_dim'''
        embed_dim = input_dim * 2

        self.layer1 = nn.Sequential(PositionEmb(input_dim), nn.Linear(embed_dim, channels), nn.Softplus())
        self.layer2 = nn.Sequential(nn.Linear(channels, 2 * channels), nn.Softplus())
        self.layer3 = nn.Sequential(nn.Linear(2 * channels, 4 * channels), nn.Softplus())
        self.layer4 = nn.Sequential(nn.Linear(4 * channels, 2 * channels), nn.Softplus())

        self.fc = nn.Sequential(nn.Linear(2 * channels, output_dim))

        self.to(device)

    def forward(self, x, y):
        emb_t_0 = self.act(self.time_fc0(y))
        emb_t_1 = self.act(self.time_fc1(emb_t_0))
        emb_t_2 = self.act(self.time_fc2(emb_t_1))
        emb_t_3 = self.act(self.time_fc3(emb_t_2))

        out = self.layer1(x) + emb_t_0
        out = self.layer2(out) + emb_t_1
        out = self.layer3(out) + emb_t_2
        out = self.layer4(out) + emb_t_3

        out = self.fc(out)

        return out


class BasicModel_emb2(nn.Module):  # Resnet
    def __init__(self, input_dim, output_dim, num_layers=1, channels=128, device='cpu'):
        super(BasicModel_emb2, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(channels),
            nn.Linear(channels, channels * 4), 
            Mish(), 
            nn.Linear(channels * 4, channels)
        )
        self.time_fc0 = nn.Linear(channels, 2 * channels)
        #self.time_fc1 = nn.Linear(channels, 2 * channels)
        self.time_fc2 = nn.Linear(channels, 4 * channels)
        #self.time_fc3 = nn.Linear(channels, 2 * channels)
        self.act = nn.SiLU()

        '''
        self.Fourier_num = [7, 8]
        self.Fourier_n = len(self.Fourier_num)
        embed_dim = (2 * self.Fourier_n + 1) * input_dim'''
        embed_dim = input_dim * 2

        self.FourierEmb = PositionEmb(input_dim)

        self.layer1_1 = nn.Sequential(nn.Linear(embed_dim, 2 * channels), Mish())
        self.layer1_2 = nn.Sequential(nn.Linear(2 * channels, 2 * channels), Mish())
        self.layer1_3 = nn.Sequential(nn.Linear(embed_dim, 2 * channels), Mish())

        #self.layer2_1 = nn.Sequential(nn.Linear(channels, 2 * channels), Mish())
        #self.layer2_2 = nn.Sequential(nn.Linear(2 * channels, 2 * channels), Mish())
        #self.layer2_3 = nn.Sequential(nn.Linear(channels, 2 * channels), Mish())

        self.midlayer = PreNorm(2 * channels, nn.Linear(2 * channels, 2 * channels))

        self.layer3_1 = nn.Sequential(nn.Linear(2 * channels, 4 * channels), Mish())
        self.layer3_2 = nn.Sequential(nn.Linear(4 * channels, 4 * channels), Mish())
        self.layer3_3 = nn.Sequential(nn.Linear(2 * channels, 4 * channels), Mish())

        #self.layer4_1 = nn.Sequential(nn.Linear(4 * channels, 2 * channels), Mish())
        #self.layer4_2 = nn.Sequential(nn.Linear(2 * channels, 2 * channels), Mish())
        #self.layer4_3 = nn.Sequential(nn.Linear(4 * channels, 2 * channels), Mish())

        self.fc = nn.Sequential(nn.Linear(4 * channels, output_dim))

        self.to(device)

    def forward(self, x, y):
        emb_t = self.time_mlp(y)

        emb_t_0 = self.act(self.time_fc0(emb_t))
        #emb_t_1 = self.act(self.time_fc1(emb_t))
        emb_t_2 = self.act(self.time_fc2(emb_t))
        #emb_t_3 = self.act(self.time_fc3(emb_t))

        embed_ins = self.FourierEmb(x)
        out = self.layer1_2(self.layer1_1(embed_ins) + emb_t_0) + self.layer1_3(embed_ins)
        #out = self.layer2_2(self.layer2_1(out) + emb_t_1) + self.layer2_3(out)
        out = self.midlayer(out) + out
        out = self.layer3_2(self.layer3_1(out) + emb_t_2) + self.layer3_3(out)
        #out = self.layer4_2(self.layer4_1(out) + emb_t_3) + self.layer4_3(out)

        out = self.fc(out)

        return out


class BNet(nn.Module):  # Resnet
    def __init__(self, input_dim, output_dim, num_layers=1, channels=128, device='cpu'):
        super(BNet, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(channels),
            nn.Linear(channels, channels * 4), 
            nn.SiLU(), 
            nn.Linear(channels * 4, channels)
        )

        self.Fourier_num = [3, 7]
        self.Fourier_n = len(self.Fourier_num)
        embed_dim = (2 * self.Fourier_n + 1) * input_dim

        self.FourierEmb = FourierPosEmb(self.Fourier_num)

        dim_mults = (1, 2)
        dims = [embed_dim, *map(lambda m: channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=channels),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=channels),
                        #Residual(PreNorm(dim_out, nn.Linear(dim_out, dim_out))),
                        #nn.Linear(dim_out, dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=channels)
        self.mid_attn = Residual(PreNorm(mid_dim, nn.Linear(mid_dim, mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=channels)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=channels),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=channels),
                        #Residual(PreNorm(dim_in, nn.Linear(dim_in, dim_in))),
                        #nn.Linear(dim_in, dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.fc = nn.Sequential(Block(channels, channels), nn.Linear(channels, output_dim))

        self.to(device)

    def forward(self, x, y):
        t = self.time_mlp(y)
        x = self.FourierEmb(x)

        h = []
        for resnet, resnet2 in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2 in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)

        return self.fc(x)


class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.softplus = nn.Softplus()

    def forward(self, input: torch.Tensor):  # type: ignore
        return input @ self.softplus(self.weight) + self.softplus(self.bias)

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def beta_linear_log_snr(t):
    return -log(torch.special.expm1(1e-4 + 10 * (t ** 2))), t

def beta_linear_log_snr_2(t):
    return -log(torch.special.expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s=0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5), t

class SNRNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        log_snr_max, log_snr_min = [beta_linear_log_snr_2(torch.tensor([time])).item() for time in (0., 1.)]
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max

        self.net = nn.Sequential(
            PositiveLinear(1, 1),
            Residual(nn.Sequential(
                PositiveLinear(1, 1024),
                nn.Sigmoid(),
                PositiveLinear(1024, 1)
                ))
            )

    def forward(self, t: torch.Tensor):  # type: ignore

        # Add start and endpoints 0 and 1.
        t = torch.cat([torch.tensor([0.0, 1.0], device=t.device), t])
        x = self.net(t[:, None])
        x = torch.squeeze(x, dim=-1)

        s0, s1, sched = x[0], x[1], x[2:]

        norm_nlogsnr = (sched - s0) / (s1 - s0)

        nlogsnr = self.intercept + self.slope * norm_nlogsnr
        return nlogsnr, norm_nlogsnr

'''
class SNRNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_min = nn.Parameter(torch.tensor(-10.0))
        self.gamma_max = nn.Parameter(torch.tensor(20.0))

        self.softplus = nn.Softplus()

    def forward(self, t: torch.Tensor):  # type: ignore

        # Add start and endpoints 0 and 1.
        t = torch.cat([torch.tensor([0.0, 1.0], device=t.device), t])
        l1 = self.l1(t[:, None])
        l2 = torch.sigmoid(self.l2(l1))
        l3 = torch.squeeze(l1 + self.l3(l2), dim=-1)

        s0, s1, sched = l3[0], l3[1], l3[2:]

        norm_nlogsnr = (sched - s0) / (s1 - s0)

        nlogsnr = self.gamma_min + self.softplus(self.gamma_max) * norm_nlogsnr
        return -nlogsnr, norm_nlogsnr'''
