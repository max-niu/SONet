from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm, LayerNorm
from models.utils import Gate
import torch
import math
from functools import partial

def scan(x0, x1, step=2):
    b, c, h, w = x1.shape
    x = torch.cat([torch.cat([x0, x1], 3), torch.cat([x1, x0], 3)], 2)
    xs = x.new_empty((b, 4, c, h*w))
    xs0 = x[:, :, ::step, ::step].contiguous()
    xs0[:, :, 1::2, :] = torch.flip(xs0[:, :, 1::2, :], dims=[-1])
    xs1 = x[:, :, 1::step, 1::step].contiguous()
    xs1[:, :, xs1.shape[2] % 2::step, :] = torch.flip(xs1[:, :, xs1.shape[2] % 2::step, :], dims=[-1])
    xs2 = x[:, :, 1::step, ::step].transpose(-1, -2).contiguous()
    xs2[:, :, 1::2, :] = torch.flip(xs2[:, :, 1::2, :], dims=[-1])
    xs3 = x[:, :, ::step, 1::step].transpose(-1, -2).contiguous()
    xs3[:, :, xs3.shape[2] % 2::2, :] = torch.flip(xs3[:, :, xs3.shape[2] % 2::2, :], dims=[-1])
    xs[:, 0] = xs0.view(b, c, -1)
    xs[:, 1] = xs1.view(b, c, -1).flip(-1)
    xs[:, 2] = xs2.view(b, c, -1)
    xs[:, 3] = xs3.view(b, c, -1).flip(-1)
    xs = xs.view(b, 4, c, -1)
    return xs

def merge(xs, h, w, step=2):
    b, k, c, l = xs.shape
    ys0 = xs[:, 0].reshape(b,c,h,w)
    ys0[:, :, 1::step, :] = torch.flip(ys0[:, :, 1::step, :], dims=[-1])
    ys1 = xs[:, 1].flip(-1).reshape(b,c,h,w)
    ys1[:, :, ys1.shape[2] % 2::2, :] = torch.flip(ys1[:, :, ys1.shape[2] % 2::2, :], dims=[-1])
    ys2 = xs[:, 2].reshape(b,c,w,h)
    ys2[:, :, 1::2, :] = torch.flip(ys2[:, :, 1::2, :], dims=[-1])
    ys3 = xs[:, 3].flip(-1).reshape(b,c,w,h)
    ys3[:, :, ys3.shape[2] % 2::2, :] = torch.flip(ys3[:, :, ys3.shape[2] % 2::2, :], dims=[-1])
    ys = torch.zeros((b, c, 2*h, 2*w), device=xs.device, dtype=xs.dtype)
    ys[:, :, ::step, ::step] = ys0
    ys[:, :, 1::step, 1::step] = ys1
    ys[:, :, 1::step, ::step] = ys2.transpose(-1, -2)
    ys[:, :, ::step, 1::step] = ys3.transpose(-1, -2)
    yt, yb = ys.chunk(2, dim=2)
    ytl, ytr = yt.chunk(2, dim=3)
    ybl, ybr = yb.chunk(2, dim=3)
    return ytl + ybr, ytr + ybl

class CrossSSM(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.ssm_0 = Mamba(dim)
        self.ssm_1 = Mamba(dim)
        self.ssm_2 = Mamba(dim)
        self.ssm_3 = Mamba(dim)

        self.norm_0 = RMSNorm(dim, eps=1e-5)
        self.norm_1 = RMSNorm(dim, eps=1e-5)
        self.norm_2 = RMSNorm(dim, eps=1e-5)
        self.norm_3 = RMSNorm(dim, eps=1e-5)

        self.apply(partial(_init_weights,n_layer=4))
        self.gate = Gate(dim, dim)

    def forward(self, xs, batch, h, w, kp):
        x0, x1 = xs.split(batch)
        x0, x1 = x0.flatten(2, 3), x1.flatten(2, 3)
        x0, x1 = x0 + kp, x1 + kp
        x0, x1 = x0.view(batch, -1, h, w), x1.view(batch, -1, h, w)

        x = scan(x0, x1).transpose(2, 3)
        x_0 = x[:, 0] + self.ssm_0(self.norm_0(x[:, 0]))
        x_1 = x[:, 1] + self.ssm_1(self.norm_1(x[:, 1]))
        x_2 = x[:, 2] + self.ssm_2(self.norm_2(x[:, 2]))
        x_3 = x[:, 3] + self.ssm_3(self.norm_3(x[:, 3]))
        x = torch.stack([x_0, x_1, x_2, x_3], 1).transpose(2, 3)

        x0, x1 = merge(x, h, w, 2)
        desc = self.gate(torch.cat([x0, x1], 0))
        x0, x1 = torch.chunk(desc, 2, dim=0)
        return x0, x1

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, torch.nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)