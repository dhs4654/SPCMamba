import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z *res + x

class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            ChannelAttention(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim, bimamba_type="v3")
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, main_feat, main_resi, guide_feat):
        main_resi = main_feat + main_resi
        main_feat = self.norm1(main_resi)
        guide_feat = self.norm2(guide_feat)

        global_f = self.cross_mamba(main_feat, extra_emb=guide_feat)

        B, HW, C = global_f.shape
        main_feat = global_f.transpose(1, 2).view(B, C, int(math.sqrt(HW)), int(math.sqrt(HW)))
        main_feat = (self.dwconv(main_feat) + main_feat).flatten(2).transpose(1, 2)

        return main_feat, main_resi


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=1, stride=1, in_chans=3, embed_dim=96, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim, 'BiasFree')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnEmbed(nn.Module):

    def __init__(self, basefilter):
        super().__init__()
        self.nc = basefilter

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])
        return x

class Attention(nn.Module):
    def __init__(
            self,
            d_model,
            window_size,
            num_clusters=16,
            d_state=8,
            d_conv=3,
            expand=2.66,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_clusters = num_clusters
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj_opt = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_sar = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d_opt = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs,
        )
        self.act_opt = nn.SiLU()

        self.conv2d_sar = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs,
        )
        self.act_sar = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm_opt = nn.LayerNorm(self.d_inner)
        self.out_norm_sar = nn.LayerNorm(self.d_inner)
        self.out_proj_opt = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_sar = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def knn_cluster_assignment(x, num_clusters):
        B, N, C = x.shape
        device = x.device
        anchor_indices = torch.stack([torch.randperm(N, device=device)[:num_clusters] for _ in range(B)])
        expanded_indices = anchor_indices.unsqueeze(-1).expand(-1, -1, C)
        anchors = torch.gather(x, 1, expanded_indices)
        dists = torch.cdist(x, anchors)
        cluster_assignments = torch.argmin(dists, dim=2)
        return cluster_assignments

    def create_patches(self, image_tensor, w, order='ltr_utd'):
        B, C, H, W = image_tensor.shape
        Hg, Wg = H // w, W // w
        assert H % w == 0 and W % w == 0, f"divide error"

        patches = image_tensor.view(B, C, Hg, w, Wg, w)

        if order == 'ltr_utd':
            patches = patches.permute(0, 1, 2, 4, 3, 5)
        elif order == 'rtl_dtu':
            patches = patches.permute(0, 1, 2, 4, 3, 5).flip(2, 3, 4, 5)
        elif order == 'utd_ltr':
            patches = patches.permute(0, 1, 4, 2, 5, 3)
        elif order == 'dtu_rtl':
            patches = patches.permute(0, 1, 4, 2, 5, 3).flip(2, 3, 4, 5)
        else:
            raise ValueError(f"Unsupported order: {order}")

        return patches.reshape(B, C, -1)

    def forward(self, optical, sar):
        x_opt = rearrange(optical, 'b c h w -> b h w c')
        x_sar = rearrange(sar, 'b c h w -> b h w c')
        B, H, W, _ = x_opt.shape

        xz_opt = self.in_proj_opt(x_opt)
        x_opt, z_opt = xz_opt.chunk(2, dim=-1)
        xz_sar = self.in_proj_sar(x_sar)
        x_sar, z_sar = xz_sar.chunk(2, dim=-1)

        x_opt = self.act_opt(self.conv2d_opt(x_opt.permute(0, 3, 1, 2).contiguous()))
        x_sar = self.act_sar(self.conv2d_sar(x_sar.permute(0, 3, 1, 2).contiguous()))

        y_opt, y_sar = self.forward_core(x_opt, x_sar)

        y_opt = y_opt.transpose(1, 2).contiguous().view(B, H, W, -1)
        y_sar = y_sar.transpose(1, 2).contiguous().view(B, H, W, -1)

        y_opt = self.out_norm_opt(y_opt) * F.silu(z_opt)
        y_sar = self.out_norm_sar(y_sar) * F.silu(z_sar)

        out_opt = self.out_proj_opt(y_opt)
        out_sar = self.out_proj_sar(y_sar)

        return rearrange(out_opt, 'b h w c -> b c h w'), rearrange(out_sar, 'b h w c -> b c h w')

    def forward_core(self, x_opt, x_sar):
        B, C, H, W = x_opt.shape
        N = H * W

        x_opt_flat = x_opt.flatten(2).transpose(1, 2)
        x_sar_flat = x_sar.flatten(2).transpose(1, 2)

        cluster_assignments = self.knn_cluster_assignment(x_opt_flat, self.num_clusters)

        sorted_indices = torch.argsort(cluster_assignments, dim=1)
        inverse_indices = torch.argsort(sorted_indices, dim=1)

        opt_shuffled_flat = torch.gather(x_opt_flat, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, C))
        sar_shuffled_flat = torch.gather(x_sar_flat, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, C))

        opt_shuffled_2d = rearrange(opt_shuffled_flat, 'b (h w) c -> b c h w', h=H, w=W)
        sar_shuffled_2d = rearrange(sar_shuffled_flat, 'b (h w) c -> b c h w', h=H, w=W)


        orders = ['ltr_utd', 'rtl_dtu', 'utd_ltr', 'dtu_rtl']
        scan_sequences = []
        for order in orders:
            opt_scanned = self.create_patches(opt_shuffled_2d, self.window_size, order)
            sar_scanned = self.create_patches(sar_shuffled_2d, self.window_size, order)

            interleaved = torch.cat([opt_scanned.unsqueeze(-1), sar_scanned.unsqueeze(-1)], dim=-1)
            interleaved = interleaved.view(B, C, -1)
            scan_sequences.append(interleaved.unsqueeze(1))

        xs = torch.cat(scan_sequences, dim=1)
        L = 2 * N
        K = 4

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True,
        ).view(B, K, -1, L)

        y_interleaved = torch.sum(out_y, dim=1)
        y_opt_shuffled = y_interleaved[:, :, 0::2] / K
        y_sar_shuffled = y_interleaved[:, :, 1::2] / K

        inverse_indices_expanded = inverse_indices.unsqueeze(1).expand(-1, C, -1)
        y_opt = torch.gather(y_opt_shuffled, 2, inverse_indices_expanded)
        y_sar = torch.gather(y_sar_shuffled, 2, inverse_indices_expanded)

        return y_opt, y_sar

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, merge=True, device=None):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge: A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, merge=True, device=None):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MMMamba(nn.Module):
    def __init__(self, dim, LayerNorm_type, window_size=4, num_clusters=16):
        super(MMMamba, self).__init__()
        self.norm_sar = LayerNorm(dim, LayerNorm_type)
        self.norm_opt = LayerNorm(dim, LayerNorm_type)
        self.norm_sar_2 = LayerNorm(dim, LayerNorm_type)
        self.norm_opt_2 = LayerNorm(dim, LayerNorm_type)

        self.attn = Attention(dim, window_size=window_size, num_clusters=num_clusters)
        self.ffn = FeedForward(dim, ffn_expansion_factor=2)

    def forward(self, x):
        opt, sar = x
        opt_f, sar_f = self.attn(self.norm_opt(opt), self.norm_sar(sar))
        opt = opt_f + opt
        sar = sar_f + sar
        opt = self.ffn(self.norm_opt_2(opt)) + opt
        sar = self.ffn(self.norm_sar_2(sar)) + sar
        return [opt, sar]


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral): normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral): normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
        self.bias = nn.Parameter(torch.zeros(torch.Size(normalized_shape)))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN: self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi

class MMMambaCR(nn.Module):
    def __init__(self, num_channels=None, base_filter=32, args=None):
        super(MMMambaCR, self).__init__()
        self.base_filter = base_filter

        self.stride = 1
        self.patch_size = 1
        self.embed_dim = self.base_filter * self.stride * self.patch_size

        self.optical_encoder = nn.Sequential(
            nn.Conv2d(13, self.base_filter, 3, 1, 1),
            HinResBlock(self.base_filter, self.base_filter),
            HinResBlock(self.base_filter, self.base_filter),
            HinResBlock(self.base_filter, self.base_filter)
        )
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(2, self.base_filter, 3, 1, 1),
            HinResBlock(self.base_filter, self.base_filter),
            HinResBlock(self.base_filter, self.base_filter),
            HinResBlock(self.base_filter, self.base_filter)
        )
        self.mm_mamba = nn.Sequential(
            MMMamba(base_filter, 'BiasFree', window_size=4, num_clusters=16),
            MMMamba(base_filter, 'BiasFree', window_size=4, num_clusters=16),
            MMMamba(base_filter, 'BiasFree', window_size=4, num_clusters=16),
            MMMamba(base_filter, 'BiasFree', window_size=4, num_clusters=16),
            MMMamba(base_filter, 'BiasFree', window_size=4, num_clusters=16)
        )

        self.optical_to_token = PatchEmbed(in_chans=self.base_filter, embed_dim=self.embed_dim)
        self.sar_to_token = PatchEmbed(in_chans=self.base_filter, embed_dim=self.embed_dim)
        self.patchunembe = PatchUnEmbed(self.base_filter)

        self.deep_fusion1 = CrossMamba(self.embed_dim)
        self.deep_fusion2 = CrossMamba(self.embed_dim)
        self.deep_fusion3 = CrossMamba(self.embed_dim)
        self.deep_fusion4 = CrossMamba(self.embed_dim)
        self.deep_fusion5 = CrossMamba(self.embed_dim)

        self.output = Refine(base_filter, 13)

    def forward(self, x):
        sar = x[:, 0:2, :, :]
        optical = x[:, 2:, :, :]

        sar_f = self.sar_encoder(sar)
        opt_f = self.optical_encoder(optical)
        b, c, h, w = opt_f.shape

        x_fused = [opt_f, sar_f]
        opt_f_fused, sar_f_fused = self.mm_mamba(x_fused)
        opt_f = self.optical_to_token(opt_f_fused)
        sar_f = self.sar_to_token(sar_f_fused)
        residual_opt_f = 0

        opt_f, residual_opt_f = self.deep_fusion1(opt_f, residual_opt_f, sar_f)
        opt_f, residual_opt_f = self.deep_fusion2(opt_f, residual_opt_f, sar_f)
        opt_f, residual_opt_f = self.deep_fusion3(opt_f, residual_opt_f, sar_f)
        opt_f, residual_opt_f = self.deep_fusion4(opt_f, residual_opt_f, sar_f)
        opt_f, residual_opt_f = self.deep_fusion5(opt_f, residual_opt_f, sar_f)

        opt_f = self.patchunembe(opt_f, (h, w))

        cloud_free_optical = self.output(opt_f) + optical

        return cloud_free_optical