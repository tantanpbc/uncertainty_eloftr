#!/usr/bin/env python3
"""
Post-hoc calibration for EfficientLoFTR uncertainty predictions.

Two methods:
  1. Temperature scaling (single scalar s): σ_cal = s * σ  (optimise val NLL)
  2. Isotonic regression: non-parametric monotone mapping σ² → σ²_cal

Regression ECE: bin matches by predicted CDF value p̂ = Φ((y-μ)/σ),
then compare bin mean p̂ to empirical fraction inside the interval.
"""
import json, math, argparse
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


# ------------------------------------------------------------------ data
def gather_predictions(ckpt_path, data_root, n_pairs=500, device='cuda'):
    """
    Load model, run on val pairs, collect (μ, log_var, gt_offset) per match.

    Returns numpy arrays:
        mu      [N, 2]   predicted sub-pixel offsets (normalised)
        log_var [N, 2]   predicted log σ²
        gt      [N, 2]   ground-truth sub-pixel offsets
    """
    from src.loftr import LoFTR
    from src.loftr.utils.full_config import full_default_cfg
    from src.config.default import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.LOFTR.MATCH_FINE.PREDICT_VAR = False   # match your training config
    cfg.freeze()

    model = LoFTR(config=full_default_cfg)
    state = torch.load(ckpt_path, map_location='cpu')
    # lightning checkpoint stores model under 'state_dict'
    sd = {k.replace('matcher.', ''): v for k, v in state.get('state_dict', state).items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    # ponytail: minimal dataset stub — replace with your actual DataLoader
    from src.datasets.megadepth import MegaDepthDataset
    from torch.utils.data import DataLoader
    ds = MegaDepthDataset(root_dir=data_root, npz_root=data_root, mode='val',
                          min_overlap_score=0.0)
    loader = DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True)

    mus, log_vars, gts = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_pairs:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            model(batch)
            if 'expec_f' in batch and 'expec_f_gt' in batch and 'expec_f_log_var' in batch:
                mu  = batch['expec_f'].cpu().numpy()
                lv  = batch['expec_f_log_var'].cpu().numpy()
                gt  = batch['expec_f_gt'].cpu().numpy()
                # filter valid gt (same mask as loss)
                valid = np.abs(gt).max(-1) < 1.0
                mus.append(mu[valid]); log_vars.append(lv[valid]); gts.append(gt[valid])

    return np.concatenate(mus), np.concatenate(log_vars), np.concatenate(gts)


# ------------------------------------------------------------------ NLL helpers
def gaussian_nll(mu, log_var, gt):
    """Mean Gaussian NLL: 0.5*(s + (y-μ)²·exp(-s))   s=log σ²"""
    r = gt - mu
    return 0.5 * (log_var + r**2 * np.exp(-log_var)).mean()


def apply_temp(log_var, log_s):
    """σ²_cal = s² · σ²  →  log σ²_cal = log_var + 2·log_s"""
    return log_var + 2.0 * log_s


# ------------------------------------------------------------------ ECE for regression
def regression_ece(mu, sigma, gt, n_bins=20):
    """
    Regression ECE for a Gaussian predictive distribution.

    Algorithm:
      1. Compute the PIT value z_i = Φ((y_i - μ_i) / σ_i)  (should be U[0,1] if calibrated)
      2. For each confidence level p in linspace(0,1):
           expected fraction = p
           empirical fraction = mean(z_i <= p)
      3. ECE = mean |empirical - expected| over n_bins

    A well-calibrated model has ECE ≈ 0.
    """
    from scipy.stats import norm
    # flatten axes: treat x and y independently
    mu_f = mu.reshape(-1);  sigma_f = sigma.reshape(-1);  gt_f = gt.reshape(-1)
    sigma_f = np.clip(sigma_f, 1e-6, None)
    z = norm.cdf((gt_f - mu_f) / sigma_f)          # PIT values

    levels = np.linspace(0, 1, n_bins + 1)[1:]     # skip 0
    empirical = np.array([(z <= p).mean() for p in levels])
    ece = np.abs(empirical - levels).mean()
    return ece, levels, empirical


# ------------------------------------------------------------------ calibration methods
def calibrate_temperature(mu, log_var, gt):
    """
    Optimise a single log-scale parameter log_s minimising val Gaussian NLL.
    σ²_cal = exp(2·log_s) · σ²
    """
    def obj(log_s):
        lv_cal = apply_temp(log_var, log_s)
        return gaussian_nll(mu, lv_cal, gt)

    res = minimize_scalar(obj, bounds=(-3, 3), method='bounded')
    return float(res.x), float(res.fun)


def calibrate_isotonic(log_var, gt, mu):
    """
    Non-parametric isotonic regression: maps predicted σ² to calibrated σ².
    Fits on the per-sample squared error as a proxy for true variance.
    """
    sigma2_pred = np.exp(log_var).reshape(-1)
    sq_err      = ((gt - mu) ** 2).reshape(-1)          # target: empirical squared error
    iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso.fit(sigma2_pred, sq_err)
    return iso


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',      required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--n_pairs',   type=int, default=500)
    ap.add_argument('--out',       default='checkpoints/calibration_meta.json')
    ap.add_argument('--method',    default='temperature', choices=['temperature', 'isotonic', 'both'])
    args = ap.parse_args()

    print("Gathering predictions …")
    mu, log_var, gt = gather_predictions(args.ckpt, args.data_root, args.n_pairs)
    sigma = np.exp(0.5 * log_var)

    nll_before = gaussian_nll(mu, log_var, gt)
    ece_before, levels, emp_before = regression_ece(mu, sigma, gt)
    print(f"Before calibration — NLL: {nll_before:.4f}  ECE: {ece_before:.4f}")

    meta = {'nll_before': nll_before, 'ece_before': ece_before}

    if args.method in ('temperature', 'both'):
        log_s, nll_after = calibrate_temperature(mu, log_var, gt)
        sigma_cal = np.exp(log_s) * sigma
        ece_after, _, emp_after = regression_ece(mu, sigma_cal, gt)
        print(f"Temperature scaling — log_s={log_s:.4f}  NLL: {nll_after:.4f}  ECE: {ece_after:.4f}")
        meta.update({'method': 'temperature', 'log_s': log_s,
                     'nll_after': nll_after, 'ece_after': ece_after})

    if args.method in ('isotonic', 'both'):
        import pickle
        iso = calibrate_isotonic(log_var, gt, mu)
        sigma2_cal = iso.predict(np.exp(log_var).reshape(-1)).reshape(log_var.shape)
        log_var_cal = np.log(np.clip(sigma2_cal, 1e-9, None))
        nll_iso = gaussian_nll(mu, log_var_cal, gt)
        ece_iso, _, _ = regression_ece(mu, np.sqrt(sigma2_cal), gt)
        print(f"Isotonic — NLL: {nll_iso:.4f}  ECE: {ece_iso:.4f}")
        iso_path = args.out.replace('.json', '_isotonic.pkl')
        with open(iso_path, 'wb') as f: pickle.dump(iso, f)
        meta.update({'isotonic_path': iso_path, 'nll_isotonic': nll_iso, 'ece_isotonic': ece_iso})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f: json.dump(meta, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()