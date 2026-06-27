#!/usr/bin/env python3
"""
Evaluate EfficientLoFTR uncertainty and produce:
  - Metrics table (MAE, RMSE, NLL, ECE) before vs after calibration
  - Per-pair visualisations: keypoint connectors + uncertainty ellipses
  - Reliability diagram
"""
import json, math, argparse, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from pathlib import Path
from scipy.stats import norm


# ------------------------------------------------------------------ model
def load_model(ckpt_path, device='cuda'):
    from src.loftr import LoFTR
    from src.loftr.utils.full_config import full_default_cfg
    model = LoFTR(config=full_default_cfg)
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = {k.replace('matcher.', ''): v for k, v in sd.get('state_dict', sd).items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


# ------------------------------------------------------------------ calibration
def load_calibration(cal_json):
    with open(cal_json) as f: meta = json.load(f)
    log_s  = meta.get('log_s', 0.0)          # temperature scalar
    iso    = None
    if 'isotonic_path' in meta:
        with open(meta['isotonic_path'], 'rb') as f: iso = pickle.load(f)
    return log_s, iso, meta


def apply_calibration(log_var, log_s, iso):
    """Return calibrated log_var using temperature then optional isotonic."""
    lv = log_var + 2.0 * log_s                          # temperature
    if iso is not None:
        s2 = np.exp(lv)
        s2_cal = iso.predict(s2.reshape(-1)).reshape(s2.shape)
        lv = np.log(np.clip(s2_cal, 1e-9, None))
    return lv


# ------------------------------------------------------------------ metrics
def compute_metrics(mu, log_var, gt):
    """
    Returns dict with MAE, RMSE, NLL (Gaussian), ECE.
    All arrays [N, 2]; metrics averaged over axes and samples.
    """
    valid = np.isfinite(gt).all(-1) & np.isfinite(mu).all(-1)
    mu, log_var, gt = mu[valid], log_var[valid], gt[valid]

    err   = gt - mu
    mae   = np.abs(err).mean()
    rmse  = np.sqrt((err**2).mean())
    # Gaussian NLL: 0.5*(s + err²·e^{-s})
    nll   = 0.5 * (log_var + err**2 * np.exp(-log_var)).mean()

    sigma = np.exp(0.5 * log_var).reshape(-1)
    z     = norm.cdf((gt - mu).reshape(-1) / np.clip(sigma, 1e-9, None))
    levels    = np.linspace(0, 1, 21)[1:]
    empirical = np.array([(z <= p).mean() for p in levels])
    ece   = np.abs(empirical - levels).mean()

    return dict(mae=mae, rmse=rmse, nll=nll, ece=ece), levels, empirical


# ------------------------------------------------------------------ ellipse helper
def draw_ellipse(ax, cx, cy, sx, sy, n_sigma=1, **kw):
    """Draw axis-aligned uncertainty ellipse at (cx,cy) with radii n_sigma*(sx,sy)."""
    e = Ellipse((cx, cy), width=2*n_sigma*sx, height=2*n_sigma*sy,
                fill=False, **kw)
    ax.add_patch(e)


# ------------------------------------------------------------------ main visualisation
def visualise_pair(img0, img1, mkpts0, mkpts1, mu_offsets, log_var, log_var_cal,
                   scale, topk=20, out_path='vis.png'):
    """
    Three-panel figure:
      Left:  keypoint connectors
      Mid:   uncertainty ellipses BEFORE calibration (patch zoom on first keypoint)
      Right: uncertainty ellipses AFTER calibration
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # panel 0: connectors
    h0, w0 = img0.shape[:2]
    combined = np.concatenate([img0, img1], axis=1)
    axes[0].imshow(combined, cmap='gray' if combined.ndim == 2 else None)
    axes[0].set_title('Top-k Matches', fontsize=10)
    k = min(topk, len(mkpts0))
    colors = plt.cm.plasma(np.linspace(0, 1, k))
    for i in range(k):
        x0, y0 = mkpts0[i]; x1, y1 = mkpts1[i]
        axes[0].plot([x0, x1 + w0], [y0, y1], '-', color=colors[i], lw=0.8, alpha=0.7)
        axes[0].plot(x0, y0, 'o', color=colors[i], ms=2)
        axes[0].plot(x1 + w0, y1, 'o', color=colors[i], ms=2)
    axes[0].axis('off')

    # panels 1 & 2: ellipses for a patch around keypoint 0
    for panel_idx, (lv, title) in enumerate([(log_var, 'Before Cal.'), (log_var_cal, 'After Cal.')]):
        ax = axes[panel_idx + 1]
        ax.set_title(title, fontsize=10)
        ax.imshow(img1, cmap='gray' if img1.ndim == 2 else None, alpha=0.6)

        for i in range(k):
            cx, cy = mkpts1[i]
            sx, sy = np.exp(0.5 * lv[i])  # σ per axis, in normalised space → pixels
            # scale from normalised [-1,1] to pixel space: multiply by half-window * scale_factor
            # lv is already in normalised coords; 1 unit ≈ (3//2 * scale) pixels
            px_scale = 1.5 * scale
            for ns, alpha, lw in [(1, 0.9, 1.2), (2, 0.5, 0.8), (3, 0.25, 0.5)]:
                draw_ellipse(ax, cx, cy, sx*px_scale*ns, sy*px_scale*ns,
                             color='cyan', lw=lw, alpha=alpha)
            # true residual vector (mu_offset is the prediction, zero gt → residual = -mu_offset)
            ax.annotate('', xy=(cx - mu_offsets[i, 0]*px_scale, cy - mu_offsets[i, 1]*px_scale),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
        ax.set_xlim(0, img1.shape[1]); ax.set_ylim(img1.shape[0], 0)
        ax.axis('off')

        legend = [mpatches.Patch(fc='none', ec='cyan', lw=1.2, label='1σ'),
                  mpatches.Patch(fc='none', ec='cyan', lw=0.8, alpha=0.5, label='2σ'),
                  mpatches.Patch(fc='none', ec='cyan', lw=0.5, alpha=0.25, label='3σ'),
                  mpatches.Patch(fc='none', ec='red', label='pred residual')]
        ax.legend(handles=legend, loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ------------------------------------------------------------------ reliability diagram
def plot_reliability(levels_before, emp_before, levels_after, emp_after, out_path):
    """
    Reliability diagram: expected fraction vs empirical fraction.
    A perfectly calibrated model sits on the diagonal.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    ax.plot(levels_before, emp_before, 'r-o', ms=4, lw=1.5, label='Before cal.')
    ax.plot(levels_after,  emp_after,  'b-o', ms=4, lw=1.5, label='After cal.')
    ax.fill_between(levels_before, levels_before, emp_before, alpha=0.15, color='red')
    ax.fill_between(levels_after,  levels_after,  emp_after,  alpha=0.15, color='blue')
    ax.set_xlabel('Expected confidence level')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Reliability Diagram')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ------------------------------------------------------------------ evaluation loop
def evaluate(model, loader, cal_json, out_dir, n_vis=5, device='cuda'):
    log_s, iso, cal_meta = load_calibration(cal_json)

    all_mu, all_lv, all_gt = [], [], []
    vis_count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            model(batch)

            if 'expec_f' not in batch or 'expec_f_gt' not in batch:
                continue

            mu  = batch['expec_f'].cpu().numpy()
            lv  = batch['expec_f_log_var'].cpu().numpy()
            gt  = batch['expec_f_gt'].cpu().numpy()
            all_mu.append(mu); all_lv.append(lv); all_gt.append(gt)

            # visualise first n_vis pairs
            if vis_count < n_vis and 'mkpts0_f' in batch:
                scale = (batch['hw0_i'][0] / batch['hw0_f'][0]).item()
                img0 = batch.get('image0', torch.zeros(1,1,256,256))[0].permute(1,2,0).cpu().numpy()
                img1 = batch.get('image1', torch.zeros(1,1,256,256))[0].permute(1,2,0).cpu().numpy()
                pts0 = batch['mkpts0_f'].cpu().numpy()
                pts1 = batch['mkpts1_f'].cpu().numpy()
                lv_cal = apply_calibration(lv, log_s, iso)
                visualise_pair(
                    img0, img1, pts0, pts1,
                    mu_offsets=mu, log_var=lv, log_var_cal=lv_cal,
                    scale=scale,
                    out_path=str(out_dir / f'pair_{i:04d}.png')
                )
                vis_count += 1

    mu  = np.concatenate(all_mu)
    lv  = np.concatenate(all_lv)
    gt  = np.concatenate(all_gt)
    lv_cal = apply_calibration(lv, log_s, iso)

    metrics_b, levels_b, emp_b = compute_metrics(mu, lv, gt)
    metrics_a, levels_a, emp_a = compute_metrics(mu, lv_cal, gt)

    print("\n=== Metrics ===")
    print(f"{'Metric':<10} {'Before':>10} {'After':>10}")
    for k in ['mae','rmse','nll','ece']:
        print(f"{k:<10} {metrics_b[k]:>10.4f} {metrics_a[k]:>10.4f}")

    plot_reliability(levels_b, emp_b, levels_a, emp_a,
                     str(out_dir / 'reliability_diagram.png'))

    results = {
        'before': {k: float(v) for k,v in metrics_b.items()},
        'after':  {k: float(v) for k,v in metrics_a.items()},
    }
    with open(out_dir / 'metrics.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"Metrics → {out_dir / 'metrics.json'}")


# ------------------------------------------------------------------ entry point
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',      required=True)
    ap.add_argument('--cal_json',  required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_dir',   default='results/uncertainty_eval')
    ap.add_argument('--n_vis',     type=int, default=5)
    ap.add_argument('--n_pairs',   type=int, default=200)
    ap.add_argument('--device',    default='cuda')
    args = ap.parse_args()

    from src.datasets.megadepth import MegaDepthDataset
    from torch.utils.data import DataLoader

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model  = load_model(args.ckpt, args.device)
    ds     = MegaDepthDataset(root_dir=args.data_root, npz_root=args.data_root,
                               mode='test', min_overlap_score=0.0)
    loader = DataLoader(ds, batch_size=1, num_workers=4,
                        pin_memory=True)

    evaluate(model, loader, args.cal_json, out_dir, args.n_vis, args.device)


if __name__ == '__main__':
    main()