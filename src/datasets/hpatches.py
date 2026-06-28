"""
HPatches dataset for EfficientLoFTR uncertainty calibration & evaluation.

HPatches structure (per sequence):
    {seq}/1.ppm           ← reference image
    {seq}/2.ppm ... 6.ppm ← query images
    {seq}/H_1_2 ... H_1_6 ← 3×3 homographies: ref pixel → query pixel (original res)

ROOT CAUSE OF "size 80 but got size 640" ERROR:
    The transformer's feature-crop optimisation (transformer.py lines 130-136)
    uses mask0.size(-2) and mask0.size(-1) as the target spatial size, then
    clips the coarse feature map (80×80 for 640-px input) using those dimensions.
    MegaDepth downsampls masks to coarse scale (1/8) via F.interpolate before
    returning them. We must do the same — returning a full-res [640,640] mask
    caused torch.cat([feat_80, zeros_640]) → size mismatch.

Fix: mask is returned at 1/8 resolution (COARSE_SCALE = 0.125), matching exactly
     what src/datasets/megadepth.py does with `self.coarse_scale = 0.125`.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

# Must match the model's backbone downsample factor (8×)
COARSE_SCALE = 0.125   # 1/8


def _read_gray(path, resize=640, df=8):
    """
    Read image, resize longest edge to `resize`, pad to square divisible by `df`.

    Returns:
        tensor      [1, pad, pad]       float32 in [0, 1]
        mask_coarse [pad//8, pad//8]    bool at COARSE (1/8) scale  ← key fix
        scale_wh    [2]                 (w_orig/w_new, h_orig/h_new)
        (nw, nh)                        resized (pre-pad) dimensions in pixels
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to read: {path}"
    h, w = img.shape

    scale = resize / max(h, w)
    nh    = (int(round(h * scale)) // df) * df
    nw    = (int(round(w * scale)) // df) * df
    img   = cv2.resize(img, (nw, nh))

    # square-pad so images in a batch can be stacked
    pad    = max(nh, nw)
    canvas = np.zeros((pad, pad), dtype=np.uint8)
    canvas[:nh, :nw] = img

    # full-res validity mask: 1 where real pixels exist, 0 in the padded region
    mask_full = np.zeros((pad, pad), dtype=np.float32)
    mask_full[:nh, :nw] = 1.0

    tensor   = torch.from_numpy(canvas).float()[None] / 255.0   # [1, pad, pad]
    scale_wh = torch.tensor([w / nw, h / nh], dtype=torch.float32)

    # Downsample mask to coarse resolution — identical to megadepth.py:
    #   F.interpolate(mask[None][None].float(), scale_factor=0.125, mode='nearest')
    mask_t      = torch.from_numpy(mask_full)[None, None]        # [1, 1, pad, pad]
    mask_coarse = F.interpolate(
        mask_t, scale_factor=COARSE_SCALE,
        mode='nearest', recompute_scale_factor=False
    )[0, 0].bool()                                               # [pad//8, pad//8]

    return tensor, mask_coarse, scale_wh, (nw, nh)


class HPatchesDataset(Dataset):
    """
    Each item = one (reference, query) pair from an HPatches sequence.
    Yields up to 5 items per sequence (query images 2–6 vs reference 1).
    """

    def __init__(self, root: str, resize: int = 640, df: int = 8,
                 sequences: str = 'all'):
        """
        Args:
            root:      path to hpatches-sequences-release/
            resize:    longest-edge resize target in pixels
            df:        divisibility factor — must match model stride (8)
            sequences: 'all' | 'i' (illumination) | 'v' (viewpoint)
        """
        self.root   = Path(root)
        self.resize = resize
        self.df     = df

        seqs = sorted(self.root.iterdir())
        if sequences == 'i':
            seqs = [s for s in seqs if s.name.startswith('i_')]
        elif sequences == 'v':
            seqs = [s for s in seqs if s.name.startswith('v_')]

        self.pairs = []
        for seq in seqs:
            ref = seq / '1.ppm'
            if not ref.exists():
                continue
            for qi in range(2, 7):
                qimg = seq / f'{qi}.ppm'
                hmat = seq / f'H_1_{qi}'
                if qimg.exists() and hmat.exists():
                    self.pairs.append((ref, qimg, hmat, seq.name, qi))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ref_path, qry_path, h_path, seq_name, qi = self.pairs[idx]

        img0, mask0, scale0, (nw0, nh0) = _read_gray(ref_path, self.resize, self.df)
        img1, mask1, scale1, (nw1, nh1) = _read_gray(qry_path, self.resize, self.df)

        # H maps pixel coords in ORIGINAL ref → ORIGINAL query (given by HPatches)
        H = np.loadtxt(str(h_path)).astype(np.float32)   # [3, 3]

        # Adjust H for our resize + padding transform:
        #   p_orig = diag(scale_wh) · p_resized
        #   p1_orig = H · p0_orig
        #   p1_resized = diag(1/scale1) · H · diag(scale0) · p0_resized
        S0    = np.diag([scale0[0].item(), scale0[1].item(), 1.0])
        S1inv = np.diag([1.0 / scale1[0].item(), 1.0 / scale1[1].item(), 1.0])
        H_resized = (S1inv @ H @ S0).astype(np.float32)

        return {
            'image0':       img0,                               # [1, pad, pad]
            'image1':       img1,                               # [1, pad, pad]
            'mask0':        mask0,                              # [pad//8, pad//8] bool
            'mask1':        mask1,                              # [pad//8, pad//8] bool
            'scale0':       scale0,                             # [2]
            'scale1':       scale1,                             # [2]
            'H_0to1':       torch.from_numpy(H_resized),        # [3, 3]
            'dataset_name': 'hpatches',
            'pair_names':   (f'{seq_name}/1.ppm', f'{seq_name}/{qi}.ppm'),
        }