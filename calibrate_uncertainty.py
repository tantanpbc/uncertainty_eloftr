#!/usr/bin/env python3
"""
Post-hoc calibration for EfficientLoFTR uncertainty predictions on HPatches.

Bugs fixed vs original:
  BUG 1: checked `if 'expec_f' in batch` at inference — fine_matching.py never
          writes data['expec_f'] (only loftr_loss does during training).
          Fixed: GT offset derived from homography H_0to1 directly.
  BUG 2: MegaDepthDataset hardcoded. Fixed: uses HPatchesDataset.
  BUG 3: supervision.py raises NotImplementedError for non-ScanNet/MegaDepth.
          Fixed: no supervision called at inference.
  BUG 4: np.concatenate([]) crash when nothing collected. Fixed: guard added.
"""
import json, argparse, sys, traceback
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────── model loading
def load_model(ckpt_path, device):
    from src.loftr import LoFTR
    from src.loftr.utils.full_config import full_default_cfg

    model = LoFTR(config=full_default_cfg)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    sd = {k.replace('matcher.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


# ─────────────────────────────────────────── GT offset from homography
def compute_expec_f_gt_from_H(mkpts0, mkpts1, H, fine_window_half):
    """
    Derive normalised sub-pixel GT offset from homography.

    Formula:
        p1_gt      = H @ [p0_x, p0_y, 1]^T   (warp ref keypoint to query space)
        residual   = p1_gt - mkpts1_f          (pixel error of model prediction)
        expec_f_gt = residual / fine_window_half   (normalise to ~[-1, 1])

    valid = |expec_f_gt| < 1.0 on both axes  (within the fine regression window)
    """
    N    = mkpts0.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    p0h  = np.concatenate([mkpts0.astype(np.float64), ones], axis=1)  # [N, 3]
    p1h  = (H.astype(np.float64) @ p0h.T).T                           # [N, 3]
    p1   = p1h[:, :2] / p1h[:, 2:3]                                   # [N, 2]

    residual   = p1 - mkpts1.astype(np.float64)
    expec_f_gt = residual / fine_window_half
    valid      = (np.abs(expec_f_gt) < 1.0).all(-1)
    return expec_f_gt.astype(np.float32), valid


# ─────────────────────────────────────────── gather predictions
def gather_predictions(ckpt_path, hpatches_root, n_pairs=500,
                       device='cuda', resize=640, sequences='all'):
    from src.datasets.hpatches import HPatchesDataset
    from torch.utils.data import DataLoader

    model  = load_model(ckpt_path, device)
    ds     = HPatchesDataset(root=hpatches_root, resize=resize, sequences=sequences)
    print(f"  Dataset: {len(ds)} pairs found in {hpatches_root}")

    loader = DataLoader(ds, batch_size=1, num_workers=0, pin_memory=False,
                        collate_fn=lambda x: x[0])

    fine_window_half = None
    mus, log_vars, gts = [], [], []
    n_skip_model  = 0
    n_skip_nokeys = 0
    n_skip_novalid = 0

    with torch.no_grad():
        for i, sample in enumerate(loader):
            if i >= n_pairs:
                break

            # add batch dim to every tensor
            batch = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(device)
                else:
                    batch[k] = v

            # ── run model ────────────────────────────────────────────────
            try:
                model(batch)
            except Exception:
                n_skip_model += 1
                if n_skip_model <= 3:
                    # print full traceback for the first 3 failures so user can debug
                    print(f"\n  [skip] pair {i} — full traceback:")
                    traceback.print_exc()
                else:
                    print(f"  [skip] pair {i} model error (run with --verbose for traceback)")
                continue

            # ── check outputs present ─────────────────────────────────────
            if 'mkpts0_f' not in batch or 'expec_f_log_var' not in batch:
                n_skip_nokeys += 1
                continue
            if batch['mkpts0_f'].shape[0] == 0:
                n_skip_nokeys += 1
                continue

            mkpts0  = batch['mkpts0_f'].cpu().numpy()         # [M, 2]
            mkpts1  = batch['mkpts1_f'].cpu().numpy()         # [M, 2]
            log_var = batch['expec_f_log_var'].cpu().numpy()  # [M, 2]

            # ── compute fine_window_half on first successful pair ─────────
            if fine_window_half is None:
                hw0_i = batch['hw0_i']
                hw0_f = batch['hw0_f']
                # hw0_i / hw0_f gives the spatial downscale factor (= 8 for default config)
                si = hw0_i[0].item() if isinstance(hw0_i[0], torch.Tensor) else hw0_i[0]
                sf = hw0_f[0].item() if isinstance(hw0_f[0], torch.Tensor) else hw0_f[0]
                fine_window_half = 1.5 * (si / sf)   # half-width of 3×3 window in pixels
                print(f"  fine_window_half = {fine_window_half:.2f} px  "
                      f"(scale factor = {si/sf:.1f})")

            # ── GT from homography ───────────────────────────────────────
            H          = batch['H_0to1'][0].cpu().numpy()
            gt, valid  = compute_expec_f_gt_from_H(mkpts0, mkpts1, H, fine_window_half)

            if valid.sum() == 0:
                n_skip_novalid += 1
                continue

            # store mu=0 and gt=residual — equivalent to (y - mu) for NLL
            mus.append(np.zeros_like(gt[valid]))
            log_vars.append(log_var[valid])
            gts.append(gt[valid])

            if (i + 1) % 50 == 0:
                n_ok = sum(len(g) for g in gts)
                print(f"  {i+1}/{min(n_pairs, len(ds))} pairs — "
                      f"{n_ok} matches  "
                      f"(skipped: {n_skip_model} model err, "
                      f"{n_skip_nokeys} no keys, {n_skip_novalid} no valid gt)")

    print(f"\n  Done. model_errors={n_skip_model}  no_keys={n_skip_nokeys}  "
          f"no_valid_gt={n_skip_novalid}  good_pairs={len(mus)}")

    if not mus:
        raise RuntimeError(
            "No valid matches collected.\n"
            "  - Check --hpatches_root points to the hpatches-sequences-release/ folder\n"
            "  - Check the checkpoint path is correct\n"
            "  - The first 3 model errors above show full tracebacks — read them carefully"
        )

    return np.concatenate(mus), np.concatenate(log_vars), np.concatenate(gts)


# ─────────────────────────────────────────── NLL / ECE helpers
def gaussian_nll(mu, log_var, gt):
    """
    Gaussian NLL: L = 0.5 * (s + (y-μ)² * exp(-s))   where s = log σ²
    """
    r = gt - mu
    return float(0.5 * (log_var + r ** 2 * np.exp(-log_var)).mean())


def apply_temp(log_var, log_s):
    """Temperature scaling: log σ²_cal = log_var + 2·log_s"""
    return log_var + 2.0 * log_s


def regression_ece(mu, sigma, gt, n_bins=20):
    """
    Regression ECE via Probability Integral Transform (PIT).
    z_i = Φ((y_i - μ_i) / σ_i) should be U[0,1] if calibrated.
    ECE = mean |empirical_fraction(z ≤ p) - p| over p in [0,1].
    """
    from scipy.stats import norm
    z         = norm.cdf((gt.reshape(-1) - mu.reshape(-1)) /
                          np.clip(sigma.reshape(-1), 1e-6, None))
    levels    = np.linspace(0, 1, n_bins + 1)[1:]
    empirical = np.array([(z <= p).mean() for p in levels])
    return float(np.abs(empirical - levels).mean()), levels, empirical


# ─────────────────────────────────────────── calibration
def calibrate_temperature(mu, log_var, gt):
    def obj(log_s):
        return gaussian_nll(mu, apply_temp(log_var, log_s), gt)
    res = minimize_scalar(obj, bounds=(-3.0, 3.0), method='bounded')
    return float(res.x), float(res.fun)


def calibrate_isotonic(log_var, gt, mu):
    iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso.fit(np.exp(log_var).reshape(-1), ((gt - mu) ** 2).reshape(-1))
    return iso


# ─────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',          required=True,  help='path to .ckpt file')
    ap.add_argument('--hpatches_root', required=True,  help='path to hpatches-sequences-release/')
    ap.add_argument('--n_pairs',       type=int, default=500)
    ap.add_argument('--resize',        type=int, default=640)
    ap.add_argument('--sequences',     default='all', choices=['all', 'i', 'v'])
    ap.add_argument('--method',        default='both',
                    choices=['temperature', 'isotonic', 'both'])
    ap.add_argument('--out',           default='checkpoints/calibration_meta.json')
    ap.add_argument('--device',        default='cuda')
    args = ap.parse_args()

    print(f"Gathering predictions from {args.n_pairs} HPatches pairs …")
    mu, log_var, gt = gather_predictions(
        args.ckpt, args.hpatches_root, args.n_pairs,
        device=args.device, resize=args.resize, sequences=args.sequences,
    )
    print(f"Collected {len(mu)} matches total.")

    sigma      = np.exp(0.5 * log_var)
    nll_before = gaussian_nll(mu, log_var, gt)
    ece_before, levels_b, emp_b = regression_ece(mu, sigma, gt)
    print(f"\nBefore calibration  — NLL: {nll_before:.4f}   ECE: {ece_before:.4f}")

    meta = {'nll_before': nll_before, 'ece_before': ece_before, 'n_matches': int(len(mu))}

    if args.method in ('temperature', 'both'):
        log_s, nll_t = calibrate_temperature(mu, log_var, gt)
        ece_t, _, _  = regression_ece(mu, np.exp(log_s) * sigma, gt)
        print(f"Temperature scaling — log_s={log_s:+.4f}   NLL: {nll_t:.4f}   ECE: {ece_t:.4f}")
        meta.update({'method': 'temperature', 'log_s': log_s,
                     'nll_after': nll_t, 'ece_after': ece_t})

    if args.method in ('isotonic', 'both'):
        import pickle
        iso    = calibrate_isotonic(log_var, gt, mu)
        s2_cal = iso.predict(np.exp(log_var).reshape(-1)).reshape(log_var.shape)
        lv_cal = np.log(np.clip(s2_cal, 1e-9, None))
        nll_iso = gaussian_nll(mu, lv_cal, gt)
        ece_iso, _, _ = regression_ece(mu, np.sqrt(s2_cal), gt)
        print(f"Isotonic regression — NLL: {nll_iso:.4f}   ECE: {ece_iso:.4f}")
        iso_path = args.out.replace('.json', '_isotonic.pkl')
        with open(iso_path, 'wb') as f:
            pickle.dump(iso, f)
        meta.update({'isotonic_path': iso_path,
                     'nll_isotonic': nll_iso, 'ece_isotonic': ece_iso})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()