import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger


# Variance clipping bounds (log-space):
#   exp(-10) ≈ 4.5e-5  (near-zero variance floor)
#   exp( 5)  ≈ 148     (variance ceiling in normalised coords)
VAR_LOG_MIN = -10.0
VAR_LOG_MAX  =  5.0


def _heatmap_log_var(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Derive per-match log-variance from the 3×3 softmax heatmap.

    The spatial variance of the probability mass is a natural measure of
    positional uncertainty — a sharp peak gives low variance, a flat
    distribution gives high variance.

    Formula (per axis independently):
        σ²_x = Σ_k  p_k · (grid_k_x - μ_x)²
        σ²_y = Σ_k  p_k · (grid_k_y - μ_y)²
        log_var = log(σ²).clamp(VAR_LOG_MIN, VAR_LOG_MAX)

    Args:
        heatmap: [M, 3, 3]  softmax probability over 3×3 grid

    Returns:
        log_var: [M, 2]  log σ² for (x, y) independently
    """
    M      = heatmap.shape[0]
    device = heatmap.device
    dtype  = heatmap.dtype

    # Grid coordinates in normalised [-1, 1] space, matching dsnt convention.
    # create_meshgrid returns [1, H, W, 2] in (x, y) order.
    grid = create_meshgrid(3, 3, normalized_coordinates=True, device=device).to(dtype)
    grid = grid.view(1, 9, 2).expand(M, -1, -1)   # [M, 9, 2]
    p    = heatmap.view(M, 9)                       # [M, 9]

    # μ per axis (same result as dsnt.spatial_expectation2d)
    mu   = (p.unsqueeze(-1) * grid).sum(dim=1)     # [M, 2]

    # spatial variance: E[(x - μ_x)²], E[(y - μ_y)²]
    diff = grid - mu.unsqueeze(1)                   # [M, 9, 2]
    var  = (p.unsqueeze(-1) * diff.pow(2)).sum(dim=1)  # [M, 2]

    var     = var.clamp(min=1e-9)
    log_var = var.log().clamp(VAR_LOG_MIN, VAR_LOG_MAX)
    return log_var                                  # [M, 2]


class FineMatching(nn.Module):
    """
    FineMatching with s2d paradigm + heteroscedastic variance output.

    New config key:
        config['match_fine']['predict_var'] (bool, default False)
            False → variance derived for free from heatmap entropy
            True  → small learned linear head on the feature slice
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_slicedim    = config['match_fine']['local_regress_slicedim']
        self.fp16     = config['half']
        self.validate = False

        self.predict_var = config['match_fine'].get('predict_var', False)
        if self.predict_var:
            slicedim = self.local_regress_slicedim
            # Learned path: pool the 9-cell feature slice, project to 2 log-var scalars.
            self.var_head = nn.Sequential(
                nn.Linear(slicedim, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 2),
            )

    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat_0 (torch.Tensor): [M, WW, C]
            feat_1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict): {
                'expec_f_log_var' (torch.Tensor): [M, 2]  log σ² per axis,
                'mkpts0_f'        (torch.Tensor): [M, 2],
                'mkpts1_f'        (torch.Tensor): [M, 2],
                ... (all original keys preserved)
            }
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: no coarse matches
        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            data.update({
                'conf_matrix_f':   torch.empty(0, WW, WW, device=feat_0.device),
                'mkpts0_f':        data['mkpts0_c'],
                'mkpts1_f':        data['mkpts1_c'],
                'expec_f_log_var': torch.empty(0, 2, device=feat_0.device),
            })
            return

        # ── compute pixel-level confidence matrices ───────────────────────────
        with torch.autocast(
            enabled=True if not (self.training or self.validate) else False,
            device_type='cuda'
        ):
            feat_f0  = feat_0[..., :-self.local_regress_slicedim]
            feat_f1  = feat_1[..., :-self.local_regress_slicedim]
            feat_ff0 = feat_0[..., -self.local_regress_slicedim:]
            feat_ff1 = feat_1[..., -self.local_regress_slicedim:]
            feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5
            conf_matrix_f  = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
            conf_matrix_ff = torch.einsum(
                'mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim)**.5
            )

        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W+2, self.W+2)
        softmax_matrix_f = softmax_matrix_f[..., 1:-1, 1:-1].reshape(M, self.WW, self.WW)

        if self.training or self.validate:
            data.update({'sim_matrix_ff': conf_matrix_ff})
            data.update({'conf_matrix_f': softmax_matrix_f})

        # ── first-stage: pixel-level match ────────────────────────────────────
        self.get_fine_ds_match(softmax_matrix_f, data)

        # ── second-stage: 3×3 local regression ───────────────────────────────
        idx_l, idx_r = data['idx_l'], data['idx_r']
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
        m_ids = m_ids[:len(data['mconf'])]
        idx_r_iids = idx_r // W
        idx_r_jids = idx_r % W

        m_ids, idx_l, idx_r_iids, idx_r_jids = (
            m_ids.reshape(-1), idx_l.reshape(-1),
            idx_r_iids.reshape(-1), idx_r_jids.reshape(-1),
        )
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long)

        m_ids      = m_ids[..., None, None].expand(-1, 3, 3)
        idx_l      = idx_l[..., None, None].expand(-1, 3, 3)
        idx_r_iids = idx_r_iids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 0]

        if idx_l.numel() == 0:
            data.update({
                'mkpts0_f':        data['mkpts0_c'],
                'mkpts1_f':        data['mkpts1_c'],
                'expec_f_log_var': torch.zeros(0, 2, device=feat_0.device),
            })
            return

        conf_matrix_ff = conf_matrix_ff.reshape(M, self.WW, self.W+2, self.W+2)
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l, idx_r_iids, idx_r_jids]  # [M, 3, 3]

        # ── variance prediction ───────────────────────────────────────────────
        if self.predict_var:
            # Learned path: pool feature slice over 3×3 neighbourhood, then project.
            # feat_ff0: [M_total, WW, slicedim] → reshape to [M_total, WW, W+2, W+2]
            feat_ff0_2d = feat_ff0.reshape(feat_ff0.shape[0], WW, W+2, W+2)
            feat_patch  = feat_ff0_2d[m_ids, idx_l, idx_r_iids, idx_r_jids]  # [M, 3, 3, slicedim]
            feat_pool   = feat_patch.reshape(-1, 9, self.local_regress_slicedim).mean(1)  # [M, slicedim]
            log_var     = self.var_head(feat_pool).clamp(VAR_LOG_MIN, VAR_LOG_MAX)        # [M, 2]
        else:
            # Free path: derive log-variance from heatmap entropy.
            # Use raw correlations before the final temperature softmax so the
            # signal is richer (the softmax for variance is computed separately).
            conf_for_var = F.softmax(
                conf_matrix_ff.reshape(-1, 9) / self.local_regress_temperature, dim=-1
            ).reshape(-1, 3, 3)
            log_var = _heatmap_log_var(conf_for_var)  # [M, 2]

        # ── coordinate expectation (original path) ────────────────────────────
        conf_matrix_ff_flat = F.softmax(
            conf_matrix_ff.reshape(-1, 9) / self.local_regress_temperature, dim=-1
        )
        heatmap           = conf_matrix_ff_flat.reshape(-1, 3, 3)
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]

        # ── scale1 for sub-pixel → pixel conversion ───────────────────────────
        if data['bs'] == 1:
            scale1 = scale * data['scale1'] if 'scale0' in data else scale
        else:
            scale1 = (
                scale * data['scale1'][data['b_ids']][:len(data['mconf']), ...]
                [:, None, :].expand(-1, -1, 2).reshape(-1, 2)
                if 'scale0' in data else scale
            )

        data.update({'expec_f_log_var': log_var})
        self.get_fine_match_local(coords_normalized, data, scale1)

    def get_fine_match_local(self, coords_normed, data, scale1):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']

        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (3 // 2) * scale1)

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
        })

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = conf_matrix.shape

        conf_matrix = conf_matrix.reshape(m, -1)[:len(data['mconf']), ...]
        val, idx    = torch.max(conf_matrix, dim=-1)
        idx         = idx[:, None]
        idx_l, idx_r = idx // WW, idx % WW

        data.update({'idx_l': idx_l, 'idx_r': idx_r})

        if self.fp16:
            grid = create_meshgrid(W, W, False, conf_matrix.device, dtype=torch.float16) - W // 2 + 0.5
        else:
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)

        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale

        if torch.is_tensor(scale0) and scale0.numel() > 1:
            mkpts0_f = (data['mkpts0_c'][:, None, :] + delta_l * scale0[:len(data['mconf']), ...][:, None, :]).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:, None, :] + delta_r * scale1[:len(data['mconf']), ...][:, None, :]).reshape(-1, 2)
        else:
            mkpts0_f = (data['mkpts0_c'][:, None, :] + delta_l * scale0).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:, None, :] + delta_r * scale1).reshape(-1, 2)

        data.update({
            "mkpts0_c": mkpts0_f,
            "mkpts1_c": mkpts1_f,
        })