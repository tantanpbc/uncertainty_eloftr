#!/usr/bin/env python3
"""
Evaluate EfficientLoFTR uncertainty on HPatches and produce:
  - Metrics: MAE, RMSE, NLL, ECE  (before vs after calibration)
  - Per-pair: keypoint connectors + 1σ/2σ/3σ uncertainty ellipses
  - Reliability diagram

Bugs fixed vs original:
  BUG 1: Used MegaDepthDataset — replaced with HPatchesDataset.
  BUG 2: Checked 'expec_f_gt' in batch — supervision never runs at inference.
          Fixed: derive GT from homography H_0to1 (same as calibrate_uncertainty.py).
  BUG 3: expec_f_log_var may be missing if M=0. Fixed: guard added.
  BUG 4: image squeeze was wrong for grayscale [1,1,H,W] → [H,W]. Fixed.
"""
import json, argparse, pickle, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────── model
def load_model(ckpt_path, device):
    from src.loftr import LoFTR
    from src.loftr.utils.full_config import full_default_cfg
    model = LoFTR(config=full_default_cfg)
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    sd = {k.replace('matcher.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


# ─────────────────────────────────────────── calibration
def load_calibration(cal_json):
    with open(cal_json) as f:
        meta = json.load(f)
    log_s = meta.get('log_s', 0.0)
    iso   = None
    if 'isotonic_path' in meta:
        with open(meta['isotonic_path'], 'rb') as f:
            iso = pickle.load(f)
    return log_s, iso, meta


def apply_calibration(log_var, log_s, iso):
    """
    Apply temperature then optional isotonic calibration.
    log σ²_cal = log_var + 2·log_s  (temperature)
    then isotonic predict on σ² if available.
    """
    lv = log_var + 2.0 * log_s
    if iso is not None:
        s2     = np.exp(lv)
        s2_cal = iso.predict(s2.reshape(-1)).reshape(s2.shape)
        lv     = np.log(np.clip(s2_cal, 1e-9, None))
    return lv


# ─────────────────────────────────────────── GT from homography
def compute_gt_residual(mkpts0, mkpts1, H, fine_window_half):
    """
    GT sub-pixel offset in normalised coords.
    residual_px = H(mkpts0) - mkpts1_f
    expec_f_gt  = residual_px / fine_window_half
    valid = |expec_f_gt| < 1.0 on both axes
    """
    N    = len(mkpts0)
    ones = np.ones((N, 1), dtype=np.float64)
    p0h  = np.concatenate([mkpts0.astype(np.float64), ones], axis=1)
    p1h  = (H.astype(np.float64) @ p0h.T).T
    p1   = p1h[:, :2] / p1h[:, 2:3]
    res  = (p1 - mkpts1.astype(np.float64)) / fine_window_half
    valid = (np.abs(res) < 1.0).all(-1)
    return res.astype(np.float32), valid


# ─────────────────────────────────────────── metrics
def compute_metrics(residuals, log_var, n_bins=20):
    """
    residuals [N, 2]: GT − prediction (already in normalised coords)
    log_var   [N, 2]: predicted log σ²
    """
    valid = np.isfinite(log_var).all(-1) & np.isfinite(residuals).all(-1)
    r  = residuals[valid]
    lv = log_var[valid]

    mae  = float(np.abs(r).mean())
    rmse = float(np.sqrt((r ** 2).mean()))
    nll  = float(0.5 * (lv + r ** 2 * np.exp(-lv)).mean())

    sigma     = np.exp(0.5 * lv).reshape(-1)
    z         = norm.cdf(r.reshape(-1) / np.clip(sigma, 1e-9, None))
    levels    = np.linspace(0, 1, n_bins + 1)[1:]
    empirical = np.array([(z <= p).mean() for p in levels])
    ece       = float(np.abs(empirical - levels).mean())

    return dict(mae=mae, rmse=rmse, nll=nll, ece=ece), levels, empirical


# ─────────────────────────────────────────── visualisation
def draw_ellipse(ax, cx, cy, rx, ry, **kw):
    ax.add_patch(Ellipse((cx, cy), width=2*rx, height=2*ry, fill=False, **kw))


def tensor_to_np_gray(t):
    """[1, 1, H, W] or [1, H, W] or [H, W] → [H, W] uint8"""
    t = t.squeeze().cpu().numpy()
    if t.max() <= 1.0:
        t = (t * 255).astype(np.uint8)
    return t


def visualise_pair(img0, img1, mkpts0, mkpts1, residuals, log_var, log_var_cal,
                   fine_window_half, topk=30, out_path='vis.png'):
    """
    Three panels:
      Left : matched keypoint connectors
      Mid  : uncertainty ellipses BEFORE calibration on img1
      Right: uncertainty ellipses AFTER calibration on img1
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    h0, w0 = img0.shape[:2]

    # panel 0: connectors
    combined = np.concatenate([img0, img1], axis=1)
    axes[0].imshow(combined, cmap='gray')
    axes[0].set_title('Top-k Matches')
    k      = min(topk, len(mkpts0))
    colors = plt.cm.plasma(np.linspace(0, 1, k))
    for j in range(k):
        x0, y0 = mkpts0[j];  x1, y1 = mkpts1[j]
        axes[0].plot([x0, x1 + w0], [y0, y1], '-', color=colors[j], lw=0.7, alpha=0.6)
        axes[0].plot(x0, y0, 'o', color=colors[j], ms=2)
        axes[0].plot(x1 + w0, y1, 'o', color=colors[j], ms=2)
    axes[0].axis('off')

    # panels 1 & 2: ellipses
    for panel_idx, (lv, title) in enumerate([
        (log_var,     'Before Calibration'),
        (log_var_cal, 'After Calibration'),
    ]):
        ax = axes[panel_idx + 1]
        ax.imshow(img1, cmap='gray', alpha=0.55)
        ax.set_title(title)

        for j in range(k):
            cx, cy = mkpts1[j]
            sx, sy = np.exp(0.5 * lv[j])          # σ in normalised space
            # convert to pixel space: 1 normalised unit = fine_window_half px
            rx = sx * fine_window_half
            ry = sy * fine_window_half
            for ns, alpha, lw in [(1, 0.9, 1.4), (2, 0.5, 0.9), (3, 0.2, 0.5)]:
                draw_ellipse(ax, cx, cy, rx*ns, ry*ns,
                             color='cyan', lw=lw, alpha=alpha)
            # residual arrow: model prediction to GT
            if j < len(residuals):
                rx_err = residuals[j, 0] * fine_window_half
                ry_err = residuals[j, 1] * fine_window_half
                ax.annotate('', xy=(cx + rx_err, cy + ry_err), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

        ax.set_xlim(0, img1.shape[1]); ax.set_ylim(img1.shape[0], 0)
        ax.axis('off')
        legend = [
            mpatches.Patch(fc='none', ec='cyan', lw=1.4, label='1σ'),
            mpatches.Patch(fc='none', ec='cyan', lw=0.9, alpha=0.5, label='2σ'),
            mpatches.Patch(fc='none', ec='cyan', lw=0.5, alpha=0.2, label='3σ'),
            mpatches.Patch(fc='none', ec='red',           label='GT residual'),
        ]
        ax.legend(handles=legend, loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out_path}")


def plot_reliability(levels_b, emp_b, levels_a, emp_a, out_path):
    """Reliability diagram: expected vs empirical coverage."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    ax.plot(levels_b, emp_b, 'r-o', ms=4, lw=1.5, label='Before cal.')
    ax.plot(levels_a,  emp_a,  'b-o', ms=4, lw=1.5, label='After cal.')
    ax.fill_between(levels_b, levels_b, emp_b, alpha=0.15, color='red')
    ax.fill_between(levels_a,  levels_a,  emp_a,  alpha=0.15, color='blue')
    ax.set_xlabel('Expected confidence level')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Reliability Diagram')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


# ─────────────────────────────────────────── evaluation loop
def evaluate(model, loader, log_s, iso, out_dir, n_vis, device):
    fine_window_half = None

    all_res, all_lv = [], []
    vis_count = 0

    with torch.no_grad():
        for i, sample in enumerate(loader):
            batch = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(device)
                else:
                    batch[k] = v

            try:
                model(batch)
            except Exception as e:
                print(f"  [skip pair {i}]: {e}")
                continue

            if 'mkpts0_f' not in batch or batch['mkpts0_f'].shape[0] == 0:
                continue
            if 'expec_f_log_var' not in batch:
                continue

            mkpts0  = batch['mkpts0_f'].cpu().numpy()
            mkpts1  = batch['mkpts1_f'].cpu().numpy()
            log_var = batch['expec_f_log_var'].cpu().numpy()
            H       = batch['H_0to1'][0].cpu().numpy()

            if fine_window_half is None:
                hw0_i = batch['hw0_i']
                hw0_f = batch['hw0_f']
                sc = (hw0_i[0] / hw0_f[0]).item() if isinstance(hw0_i, torch.Tensor) \
                     else hw0_i[0] / hw0_f[0]
                fine_window_half = 1.5 * sc
                print(f"fine_window_half = {fine_window_half:.2f} px")

            residuals, valid = compute_gt_residual(mkpts0, mkpts1, H, fine_window_half)
            if valid.sum() == 0:
                continue

            all_res.append(residuals[valid])
            all_lv.append(log_var[valid])

            # visualise
            if vis_count < n_vis:
                img0 = tensor_to_np_gray(batch['image0'][0])
                img1 = tensor_to_np_gray(batch['image1'][0])
                lv_cal = apply_calibration(log_var[valid], log_s, iso)
                visualise_pair(
                    img0, img1,
                    mkpts0[valid], mkpts1[valid],
                    residuals[valid], log_var[valid], lv_cal,
                    fine_window_half,
                    out_path=str(out_dir / f'pair_{i:04d}.png')
                )
                vis_count += 1

    if not all_res:
        print("ERROR: no valid matches collected — check checkpoint and data path.")
        return

    res = np.concatenate(all_res)
    lv  = np.concatenate(all_lv)
    lv_cal = apply_calibration(lv, log_s, iso)

    m_b, levels_b, emp_b = compute_metrics(res, lv)
    m_a, levels_a, emp_a = compute_metrics(res, lv_cal)

    print(f"\n{'Metric':<10} {'Before':>10} {'After':>10}")
    for k in ['mae', 'rmse', 'nll', 'ece']:
        print(f"{k:<10} {m_b[k]:>10.4f} {m_a[k]:>10.4f}")

    plot_reliability(levels_b, emp_b, levels_a, emp_a,
                     str(out_dir / 'reliability_diagram.png'))

    results = {'before': {k: float(v) for k, v in m_b.items()},
               'after':  {k: float(v) for k, v in m_a.items()}}
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics → {out_dir / 'metrics.json'}")


# ─────────────────────────────────────────── entry point
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',           required=True)
    ap.add_argument('--cal_json',       required=True)
    ap.add_argument('--hpatches_root',  required=True,
                    help='path to hpatches-sequences-release/')
    ap.add_argument('--out_dir',        default='results/uncertainty_eval')
    ap.add_argument('--n_vis',          type=int, default=5)
    ap.add_argument('--n_pairs',        type=int, default=200)
    ap.add_argument('--resize',         type=int, default=640)
    ap.add_argument('--sequences',      default='all', choices=['all', 'i', 'v'])
    ap.add_argument('--device',         default='cuda')
    args = ap.parse_args()

    from src.datasets.hpatches import HPatchesDataset
    from torch.utils.data import DataLoader

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model          = load_model(args.ckpt, args.device)
    log_s, iso, _  = load_calibration(args.cal_json)
    ds             = HPatchesDataset(root=args.hpatches_root,
                                     resize=args.resize,
                                     sequences=args.sequences)
    # collate_fn=None because images may differ in size — use batch_size=1
    loader = DataLoader(ds, batch_size=1, num_workers=2,
                        collate_fn=lambda x: x[0])

    # limit to n_pairs
    from itertools import islice
    loader = islice(loader, args.n_pairs)

    evaluate(model, loader, log_s, iso, out_dir, args.n_vis, args.device)


if __name__ == '__main__':
    main()