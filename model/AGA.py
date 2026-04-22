import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.nn.init import trunc_normal_


class DePE(nn.Module):
    def __init__(self, dim_in, dim_out, k_sizes=[3, 5], conv_op=nn.Conv2d, groups=None):
        super().__init__()
        self.groups = dim_out if groups is None else groups

        self.proj_convs = nn.ModuleList()
        for k_size in k_sizes:
            self.proj_convs.append(
                conv_op(dim_out, dim_out, k_size, 1, k_size // 2, groups=self.groups)
            )

        self.input_conv = conv_op(dim_in, dim_out, 1, 1, 0) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        x = self.input_conv(x)
        for proj in self.proj_convs:
            x = x + proj(x)
        return x


class DAttentionBaseline(nn.Module):
    def __init__(
        self,
        q_size,
        kv_size,
        n_heads,
        n_head_channels,
        n_groups,
        attn_drop,
        proj_drop,
        stride,
        offset_range_factor,
        use_pe,
        dwc_pe,
        no_off,
        fixed_pe,
        ksize,
        log_cpb,
        text_dim=512,
        text_linear_range=0.10,
        text_shift_range=0.10,
        alpha_init_bias=-2.0,
    ):
        super().__init__()

        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        if kv_size is None:
            self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        else:
            self.kv_h, self.kv_w = kv_size

        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        assert self.nc % self.n_groups == 0, 'nc must be divisible by n_groups'
        assert self.n_heads % self.n_groups == 0, 'n_heads must be divisible by n_groups'
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride

        self.text_dim = text_dim
        self.text_linear_range = text_linear_range
        self.text_shift_range = text_shift_range

        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.n_group_channels,
                self.n_group_channels,
                kk,
                stride,
                pad_size,
                groups=self.n_group_channels,
            ),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc
                )
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False),
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

        # Change 1: 4-parameter affine -> 6-parameter affine (global text field).
        self.text_to_affine_params = nn.Linear(self.text_dim, 6)
        # Change 2: scalar alpha -> per-group alpha.
        self.text_to_alpha = nn.Linear(self.text_dim + self.nc, self.n_groups)
        # self.text_to_alpha = nn.Sequential(
        #     nn.Linear(self.text_dim + self.nc, self.n_groups),
        #     nn.GELU(),
        #     nn.LayerNorm(self.n_groups)
        # )

        self._reset_text_branch(alpha_init_bias=alpha_init_bias)

    # def _reset_text_branch(self, alpha_init_bias=-2.0):
    #     nn.init.zeros_(self.text_to_affine_params.weight)
    #     nn.init.zeros_(self.text_to_affine_params.bias)
    #     nn.init.zeros_(self.text_to_alpha[0].weight)
    #     nn.init.constant_(self.text_to_alpha[0].bias, alpha_init_bias)
    def _reset_text_branch(self, alpha_init_bias=-2.0):
        nn.init.zeros_(self.text_to_affine_params.weight)
        nn.init.zeros_(self.text_to_affine_params.bias)
        nn.init.zeros_(self.text_to_alpha.weight)
        nn.init.constant_(self.text_to_alpha.bias, alpha_init_bias)
    @staticmethod
    def _safe_divisor(v: int) -> float:
        return float(max(v - 1, 1))

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        # Use the same convention as grid_sample(..., align_corners=True).
        ref_y, ref_x = torch.meshgrid(
            torch.arange(H_key, dtype=dtype, device=device),
            torch.arange(W_key, dtype=dtype, device=device),
            indexing='ij',
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(self._safe_divisor(W_key)).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(self._safe_divisor(H_key)).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(H, dtype=dtype, device=device),
            torch.arange(W, dtype=dtype, device=device),
            indexing='ij',
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(self._safe_divisor(W)).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(self._safe_divisor(H)).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def _build_text_delta(self, reference, text_feat):
        """
        Build a global 6-DoF text-induced affine field, then repeat it across groups.
        reference: (B*g, Hk, Wk, 2), where [..., 0] is y and [..., 1] is x.
        returns:
            text_delta: (B*g, Hk, Wk, 2)
        """
        B = text_feat.shape[0]
        # Keep the original design philosophy here: text gives a shared global geometric prior.
        ref_B = einops.rearrange(reference, '(b g) h w p -> b g h w p', b=B, g=self.n_groups)[:, 0, ...]
        ref_B = ref_B.clamp(-1.0, 1.0)

        da11, da12, da21, da22, tx_raw, ty_raw = self.text_to_affine_params(text_feat).chunk(6, dim=1)

        # Residual-to-identity parameterization for stable training.
        a11 = 1.0 + self.text_linear_range * torch.tanh(da11).reshape(B, 1, 1)
        a12 = self.text_linear_range * torch.tanh(da12).reshape(B, 1, 1)
        a21 = self.text_linear_range * torch.tanh(da21).reshape(B, 1, 1)
        a22 = 1.0 + self.text_linear_range * torch.tanh(da22).reshape(B, 1, 1)
        tx = self.text_shift_range * torch.tanh(tx_raw).reshape(B, 1, 1)
        ty = self.text_shift_range * torch.tanh(ty_raw).reshape(B, 1, 1)

        # Internally, [..., 0] is y and [..., 1] is x.
        y = ref_B[..., 0]
        x = ref_B[..., 1]

        x_new = a11 * x + a12 * y + tx
        y_new = a21 * x + a22 * y + ty

        text_warping_field = torch.stack((y_new, x_new), dim=-1).clamp(-1.0, 1.0)
        text_delta = text_warping_field - ref_B
        text_delta = einops.repeat(text_delta, 'b h w p -> (b g) h w p', g=self.n_groups)
        return text_delta

    def _build_alpha(self, q, text_feat):
        """
        Per-group alpha: each visual group decides how much to trust the shared text prior.
        returns alpha of shape (B*g, 1, 1, 1)
        """
        B, C, _, _ = q.shape
        q_global = q.mean(dim=(2, 3)).reshape(B, C)
        fused_global_feat = torch.cat((q_global, text_feat), dim=1)
        alpha = torch.sigmoid(self.text_to_alpha(fused_global_feat))  # (B, g)
        alpha = alpha.view(B, self.n_groups, 1, 1, 1)
        alpha = alpha.reshape(B * self.n_groups, 1, 1, 1)
        return alpha

    def forward(self, x, text_feat, return_debug=False):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(
            q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels
        )
        offset = self.conv_offset(q_off).contiguous()
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.no_off:
            offset = torch.zeros_like(offset)
        elif self.offset_range_factor >= 0:
            offset_range = torch.tensor(
                [1.0 / self._safe_divisor(Hk), 1.0 / self._safe_divisor(Wk)],
                device=device,
                dtype=dtype,
            ).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        img_pos = offset + reference

        text_delta = self._build_text_delta(reference, text_feat)
        alpha = self._build_alpha(q, text_feat)

        # Text acts as a geometric bias on top of image-driven local offsets.
        pos = img_pos + alpha * text_delta
        pos = pos.clamp(-1.0, 1.0)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f'Size is {x_sampled.size()}'
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],
                mode='bilinear',
                align_corners=True,
            )

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(
                    B * self.n_heads, self.n_head_channels, H * W
                )
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                    q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2)
                    - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
                ).mul(4.0)
                displacement = (
                    torch.sign(displacement)
                    * torch.log2(torch.abs(displacement) + 1.0)
                    / np.log2(8.0)
                )
                attn_bias = self.rpe_table(displacement)
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                    q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2)
                    - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
                ).mul(0.5)

                attn_bias = F.grid_sample(
                    input=einops.rearrange(
                        rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups
                    ),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear',
                    align_corners=True,
                )
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        if return_debug:
            debug = {
                'alpha': alpha.reshape(B, self.n_groups, 1, 1, 1),
                'text_delta': text_delta.reshape(B, self.n_groups, Hk, Wk, 2),
            }
            return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2), debug

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class TransformerMLP(nn.Module):
    def __init__(self, channels, expansion, drop):
        super().__init__()
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential(
            nn.Linear(self.dim1, self.dim2),
            nn.GELU(),
            nn.Dropout(drop, inplace=True),
            nn.Linear(self.dim2, self.dim1),
            nn.Dropout(drop, inplace=True),
        )

    def forward(self, x):
        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class TransformerMLPWithConv(nn.Module):
    def __init__(self, channels, expansion, drop):
        super().__init__()
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(nn.Conv2d(self.dim1, self.dim2, 1, 1, 0))
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Sequential(nn.Conv2d(self.dim2, self.dim1, 1, 1, 0))
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = torch.randn(1, 64, 64, 64, device=device)
    text = torch.randn(1, 512, device=device)

    net = DAttentionBaseline(
        q_size=(64, 64),
        kv_size=(16, 16),
        n_heads=4,
        n_head_channels=16,
        n_groups=4,
        attn_drop=0.0,
        proj_drop=0.0,
        stride=4,
        offset_range_factor=-1,
        use_pe=True,
        dwc_pe=False,
        no_off=False,
        fixed_pe=False,
        ksize=5,
        log_cpb=False,
    ).to(device)

    out, pos, ref, debug = net(image, text, return_debug=True)
    print('out:', out.shape)
    print('pos:', pos.shape)
    print('ref:', ref.shape)
    print('alpha:', debug['alpha'].shape)
    print('text_delta:', debug['text_delta'].shape)
