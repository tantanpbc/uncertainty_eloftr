import math

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config      = config
        self.loss_config = config['loftr']['loss']
        self.match_type       = 'dual_softmax'
        self.sparse_spvs      = self.config['loftr']['match_coarse']['sparse_spvs']
        self.fine_sparse_spvs = self.config['loftr']['match_fine']['sparse_spvs']

        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w     = self.loss_config['pos_weight']
        self.c_neg_w     = self.loss_config['neg_weight']

        # overlap weighting
        self.overlap_weightc = self.config['loftr']['loss']['coarse_overlap_weight']
        self.overlap_weightf = self.config['loftr']['loss']['fine_overlap_weight']

        # sub-pixel regression
        self.local_regressw            = self.config['loftr']['fine_window_size']
        self.local_regress_temperature = self.config['loftr']['match_fine']['local_regress_temperature']

        # 'l2' (original) | 'nll_gaussian' | 'nll_laplace'
        self.local_loss_type = self.loss_config.get('local_loss_type', 'l2')

    # ------------------------------------------------------------------ coarse
    def compute_coarse_loss(self, conf, conf_gt, weight=None, overlap_weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        del conf_gt
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w

        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'focal':
            conf  = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:
                pos_conf = conf[pos_mask]
                loss_pos = -alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                if self.overlap_weightc:
                    loss_pos = loss_pos * overlap_weight
                loss = c_pos_w * loss_pos.mean()
                return loss
            else:
                loss_pos = -alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = -alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                logger.info("conf_pos_c: {loss_pos}, conf_neg_c: {loss_neg}".format(
                    loss_pos=conf[pos_mask].mean(), loss_neg=conf[neg_mask].mean()))
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                if self.overlap_weightc:
                    loss_pos = loss_pos * overlap_weight
                loss_pos_mean = loss_pos.mean()
                loss_neg_mean = loss_neg.mean()
                logger.info("conf_pos_c: {loss_pos}, conf_neg_c: {loss_neg}".format(
                    loss_pos=conf[pos_mask].mean(), loss_neg=conf[neg_mask].mean()))
                return c_pos_w * loss_pos_mean + c_neg_w * loss_neg_mean
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))

    # ------------------------------------------------------------------ fine pixel
    def compute_fine_loss(self, conf_matrix_f, conf_matrix_f_gt, overlap_weight=None):
        """
        Args:
            conf_matrix_f     (torch.Tensor): [m, WW, WW]
            conf_matrix_f_gt  (torch.Tensor): [m, WW, WW]
        """
        if conf_matrix_f_gt.shape[0] == 0:
            if self.training:
                logger.warning("assign a false supervision to avoid ddp deadlock")
                pass
            else:
                return None

        pos_mask, neg_mask = conf_matrix_f_gt == 1, conf_matrix_f_gt == 0
        del conf_matrix_f_gt
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w

        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            c_neg_w = 0.

        conf  = torch.clamp(conf_matrix_f, 1e-6, 1-1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']

        if self.fine_sparse_spvs:
            loss_pos = -alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            if self.overlap_weightf:
                loss_pos = loss_pos * overlap_weight
            return c_pos_w * loss_pos.mean()
        else:
            loss_pos = -alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = -alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            logger.info("conf_pos_f: {loss_pos}, conf_neg_f: {loss_neg}".format(
                loss_pos=conf[pos_mask].mean(), loss_neg=conf[neg_mask].mean()))
            if self.overlap_weightf:
                loss_pos = loss_pos * overlap_weight
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

    # ------------------------------------------------------------------ sub-pixel L2 (original)
    def _compute_local_loss_l2(self, expec_f, expec_f_gt):
        """
        Standard L2 regression loss on sub-pixel coordinate offsets.

        Args:
            expec_f    [M, 2]  predicted normalised offsets (from dsnt, range ≈ [-1, 1])
            expec_f_gt [M, 2]  ground-truth normalised offsets
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    # ------------------------------------------------------------------ sub-pixel NLL (new)
    def _compute_local_loss_nll(self, expec_f, log_var, expec_f_gt, dist='gaussian'):
        """
        Heteroscedastic Negative Log-Likelihood loss for sub-pixel coordinate regression.

        Let  s = log σ²  (predicted, already clipped to [VAR_LOG_MIN, VAR_LOG_MAX])
             μ = predicted coordinate offset
             y = ground-truth coordinate offset

        ── Gaussian NLL (per axis, summed over x and y) ──────────────────────
            -log p(y|μ,σ²) = 0.5 · log(2π) + 0.5 · log σ² + 0.5 · (y-μ)²/σ²
            Dropping the constant:
            L_gauss = 0.5 · (s  +  (y-μ)² · exp(-s))

        ── Laplace NLL (per axis, summed over x and y) ───────────────────────
            For Laplace(μ, b):  σ² = 2b²  →  log b = (log σ² - log 2) / 2
            -log p(y|μ,b) = log(2b) + |y-μ|/b
                          = log 2 + log b + |y-μ| · exp(-log b)
            L_laplace = log 2 + (s - log 2)/2  +  |y-μ| · exp(-(s - log 2)/2)

        Args:
            expec_f    [M, 2]  predicted μ
            log_var    [M, 2]  predicted log σ² per axis (already clipped)
            expec_f_gt [M, 2]  ground-truth offsets
            dist       str     'gaussian' | 'laplace'

        Returns:
            scalar loss (mean over valid matches, sum over x/y axes)
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None

        mu       = expec_f[correct_mask]      # [N, 2]
        s        = log_var[correct_mask]      # [N, 2]  log σ²
        y        = expec_f_gt[correct_mask]   # [N, 2]
        residual = y - mu                     # [N, 2]

        if dist == 'gaussian':
            # L = 0.5 * (s  +  residual² · e^{-s})
            loss_per = 0.5 * (s + residual.pow(2) * (-s).exp())   # [N, 2]

        elif dist == 'laplace':
            # log b = (s - log2) / 2
            # L = log2 + log b  +  |residual| · e^{-log b}
            log_b    = (s - math.log(2.0)) / 2.0
            loss_per = math.log(2.0) + log_b + residual.abs() * (-log_b).exp()  # [N, 2]

        else:
            raise ValueError(f'Unknown dist: {dist}. Choose "gaussian" or "laplace".')

        # sum over (x, y) axes, mean over matches
        return loss_per.sum(-1).mean()

    # ------------------------------------------------------------------ c_weight
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None])
        else:
            c_weight = None
        return c_weight

    # ------------------------------------------------------------------ forward
    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}

        # 0. element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        if self.overlap_weightc:
            loss_c = self.compute_coarse_loss(
                data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn'
                    else data['conf_matrix'],
                data['conf_matrix_gt'],
                weight=c_weight,
                overlap_weight=data['conf_matrix_error_gt'])
        else:
            loss_c = self.compute_coarse_loss(
                data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn'
                    else data['conf_matrix'],
                data['conf_matrix_gt'],
                weight=c_weight)

        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. pixel-level loss (first-stage refinement)
        if self.overlap_weightf:
            loss_f = self.compute_fine_loss(
                data['conf_matrix_f'], data['conf_matrix_f_gt'],
                data['conf_matrix_f_error_gt'])
        else:
            loss_f = self.compute_fine_loss(data['conf_matrix_f'], data['conf_matrix_f_gt'])

        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})

        # 3. sub-pixel loss (second-stage refinement)
        # Build expec_f from sim_matrix_ff if not already computed by the forward pass
        if 'expec_f' not in data:
            sim_matrix_f = data['sim_matrix_ff']
            m_ids, i_ids = data['m_ids_f'], data['i_ids_f']
            j_ids_di, j_ids_dj = data['j_ids_f_di'], data['j_ids_f_dj']
            del data['sim_matrix_ff'], data['m_ids_f'], data['i_ids_f'], \
                data['j_ids_f_di'], data['j_ids_f_dj']

            delta = create_meshgrid(3, 3, True, sim_matrix_f.device).to(torch.long)
            m_ids    = m_ids[..., None, None].expand(-1, 3, 3)
            i_ids    = i_ids[..., None, None].expand(-1, 3, 3)
            j_ids_di = j_ids_di[..., None, None].expand(-1, 3, 3) + delta[None, ..., 1]
            j_ids_dj = j_ids_dj[..., None, None].expand(-1, 3, 3) + delta[None, ..., 0]

            W = self.local_regressw
            sim_matrix_f = sim_matrix_f.reshape(-1, W*W, W+2, W+2)
            sim_matrix_f = sim_matrix_f[m_ids, i_ids, j_ids_di, j_ids_dj].reshape(-1, 9)
            sim_matrix_f = F.softmax(sim_matrix_f / self.local_regress_temperature, dim=-1)
            heatmap      = sim_matrix_f.reshape(-1, 3, 3)

            coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
            data.update({'expec_f': coords_normalized})

        # Choose loss type based on config
        if self.local_loss_type == 'l2':
            loss_l = self._compute_local_loss_l2(data['expec_f'], data['expec_f_gt'])

        elif self.local_loss_type in ('nll_gaussian', 'nll_laplace'):
            dist    = 'gaussian' if 'gaussian' in self.local_loss_type else 'laplace'
            log_var = data.get('expec_f_log_var')
            if log_var is None:
                logger.warning(
                    "local_loss_type='{}' but 'expec_f_log_var' is missing in data dict. "
                    "Falling back to L2. Check that FineMatching is the modified version."
                    .format(self.local_loss_type)
                )
                loss_l = self._compute_local_loss_l2(data['expec_f'], data['expec_f_gt'])
            else:
                loss_l = self._compute_local_loss_nll(
                    data['expec_f'], log_var, data['expec_f_gt'], dist=dist
                )
        else:
            raise ValueError(
                f"Unknown local_loss_type: '{self.local_loss_type}'. "
                "Choose 'l2', 'nll_gaussian', or 'nll_laplace'."
            )

        loss += loss_l * self.loss_config['local_weight']
        loss_scalars.update({"loss_l": loss_l.clone().detach().cpu()})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})