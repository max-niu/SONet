import torch
import math
from tree_scan import _C
from models.utils import distance, bidx_edge
from torch.autograd import Function
from timm.models.layers import trunc_normal_, DropPath
from torch.autograd.function import once_differentiable
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F

class Layer(nn.Module):
    def __init__(
        self,
        indim,
        hiddendim,
        outdim,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(indim, eps=1e-6)
        self.norm2 = nn.LayerNorm(indim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scan = TreeScan(d_model=indim)
        self.mlp = nn.Sequential(
            nn.Linear(indim, hiddendim),
            nn.GELU(),
            nn.Linear(hiddendim, outdim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.scan(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None
mst = _MST.apply

class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2), col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index
    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight
    def _build_feature_weight_cosine(self, fm, max_tree):
        batch, dim = fm.shape[0], fm.shape[1]
        weight_row = torch.cosine_similarity(
            fm[:, :, :-1, :].reshape(batch, dim, -1),
            fm[:, :, 1:, :].reshape(batch, dim, -1),
            dim=1,
        )
        weight_col = torch.cosine_similarity(
            fm[:, :, :, :-1].reshape(batch, dim, -1),
            fm[:, :, :, 1:].reshape(batch, dim, -1),
            dim=1,
        )
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(
                    weight
                )
            else:
                weight = self.mapping_func(
                    -weight
                )
        return weight
    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_cosine(guide_in, max_tree)
            else:
                weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
        return tree

class FuseSSM(nn.Module):
    def __init__(self, dim, rat=4):
        super().__init__()
        self.layer1 = Layer(indim=dim, hiddendim=dim*rat, outdim=dim, drop_path=0.4722222089767456)
        self.layer2 = Layer(indim=dim, hiddendim=dim*rat, outdim=dim, drop_path=0.5)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(_init_weights)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.norm(x)
        return x

def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class _BFS(Function):
    @staticmethod
    def forward(ctx,edge_index,max_adj_per_vertex):
        sorted_index,sorted_parent, sorted_child = _C.bfs_forward(edge_index,max_adj_per_vertex)
        return sorted_index,sorted_parent,sorted_child

class _Refine(Function):
    @staticmethod
    def forward(ctx,feature_in,edge_weight,sorted_index,sorted_parent,sorted_child,edge_coef):
        (feature_aggr,feature_aggr_up) = _C.tree_scan_refine_forward(feature_in,edge_weight,sorted_index,sorted_parent,sorted_child,edge_coef)
        ctx.save_for_backward(feature_in,edge_weight,sorted_index,sorted_parent,sorted_child,feature_aggr,feature_aggr_up,edge_coef)
        return feature_aggr

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        (feature_in,edge_weight,sorted_index,sorted_parent,sorted_child,feature_aggr,feature_aggr_up,edge_coef) = ctx.saved_tensors
        grad_feature = _C.tree_scan_refine_backward_feature(feature_in,edge_weight,sorted_index,sorted_parent,sorted_child,feature_aggr,feature_aggr_up,grad_output,edge_coef)
        grad_edge_weight = _C.tree_scan_refine_backward_edge_weight(feature_in,edge_weight,sorted_index,sorted_parent,sorted_child,feature_aggr,feature_aggr_up,grad_output,edge_coef)
        return grad_feature, grad_edge_weight, None, None, None, None

def core(xs, dts, As, Bs, Cs, Ds, delta_bias, origin_shape, h_norm):
    K = 1
    _, _, H, W = origin_shape
    B, D, L = xs.shape
    dts = F.softplus(dts + delta_bias.unsqueeze(0).unsqueeze(-1))
    deltaA = (dts * As.unsqueeze(0)).exp_()
    deltaB = rearrange(dts, "b (k d) l -> b k d l", k=K, d=int(D / K)) * Bs
    BX = deltaB * rearrange(xs, "b (k d) l -> b k d l", k=K, d=int(D / K))
    bfs = _BFS.apply
    refine = _Refine.apply
    feat_in = BX.view(B, -1, L)
    edge_weight = deltaA
    fea4tree_hw = rearrange(xs, "b d (h w) -> b d h w", h=H, w=W)
    mst_layer = MinimumSpanningTree("Cosine", torch.exp)
    tree = mst_layer(fea4tree_hw)
    sorted_index, sorted_parent, sorted_child = bfs(tree, 4)
    edge_weight = bidx_edge(edge_weight, sorted_index)
    edge_weight_coef = torch.ones_like(sorted_index, dtype=edge_weight.dtype)
    feature_out = refine(
        feat_in,
        edge_weight,
        sorted_index,
        sorted_parent,
        sorted_child,
        edge_weight_coef,
    )
    if h_norm is not None:
        out = h_norm(feature_out.transpose(-1, -2).contiguous())
    y = (
        rearrange(out, "b l (k d) -> b l k d", k=K, d=int(D / K)).unsqueeze(-1)
        @ rearrange(Cs, "b k n l -> b l k n").unsqueeze(-1)
    ).squeeze(-1)
    y = rearrange(y, "b l k d -> b (k d) l")
    y = y + Ds.reshape(1, -1, 1) * xs
    return y

def scanning(
    x: torch.Tensor = None,
    x_proj_weight: torch.Tensor = None,
    dt_projs_weight: torch.Tensor = None,
    dt_projs_bias: torch.Tensor = None,
    A_logs: torch.Tensor = None,
    Ds: torch.Tensor = None,
    out_norm: torch.nn.Module = None,
    h_norm=None,
):
    B, D, H, W = x.shape
    origin_shape = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W
    xs = rearrange(x.unsqueeze(1), "b k d h w -> b k d (h w)")
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    xs = xs.to(torch.float)
    dts = dts.to(torch.float)
    Bs = Bs.to(torch.float)
    Cs = Cs.to(torch.float)
    ys = core(xs, dts, As, Bs, Cs, Ds, delta_bias, origin_shape, h_norm).view(B, K, -1, H, W)
    y = rearrange(ys, "b k d h w -> b (k d) (h w)")
    y = y.transpose(dim0=1, dim1=2).contiguous()
    y = out_norm(y).view(B, H, W, -1)
    return y.to(x.dtype)


class TreeScan(nn.Module):
    def __init__(
        self,
        d_model,
    ):
        super().__init__()
        self.d_inner = 2 * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.h_norm = nn.LayerNorm(self.d_inner)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False, device=None, dtype=None)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=False,
            kernel_size=3,
            padding=1,
            device=None,
            dtype=None,
        )
        self.x_weight = nn.Parameter(nn.Linear(self.d_inner, (self.dt_rank + 2), bias=False, device=None, dtype=None).weight.data.unsqueeze(0))
        self.out = nn.Linear(self.d_inner, d_model, bias=False, device=None, dtype=None)
        dt = self.dt_init(self.dt_rank, self.d_inner)
        self.dt_weight = nn.Parameter(dt.weight.data.unsqueeze(0))
        self.dt_bias = nn.Parameter(dt.bias.data.unsqueeze(0))
        self.a_logs = self.A_log_init(self.d_inner)
        self.ds = self.D_init(self.d_inner)

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, device=None, dtype=None)
        dt_init_std = dt_rank**-0.5 * dt_scale
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, device=None, dtype=None)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj
    @staticmethod
    def A_log_init(d_inner):
        A = repeat(
            torch.arange(1, 2, dtype=torch.float32, device=None),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        A_log = repeat(A_log, "d n -> r d n", r=1)
        A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log
    @staticmethod
    def D_init(d_inner):
        D = torch.ones(d_inner, device=None)
        D = repeat(D, "n1 -> r n1", r=1)
        D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.act(x)
        y = scanning(
            x,
            self.x_weight,
            self.dt_weight,
            self.dt_bias,
            self.a_logs,
            self.ds,
            out_norm=getattr(self, "out_norm", None),
            h_norm=self.h_norm,
        )
        y = y * z
        out = self.out(y)
        return out
