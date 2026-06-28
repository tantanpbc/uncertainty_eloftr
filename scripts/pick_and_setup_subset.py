"""
Run this once, locally, after you've downloaded data/megadepth/index/.
Picks the 20 richest scenes (most training pairs = most diverse images/views)
out of train_list.txt, excluding scenes known to have low-quality reconstructions
(reported in prior work: https://arxiv.org/pdf/2501.07556 supplementary).
Writes a filtered train_list_subset20.txt and prints the depth-download command.
"""
import numpy as np

N = 20
LIST_PATH = "data/megadepth/index/trainvaltest_list/train_list.txt"
NPZ_ROOT = "data/megadepth/index/scene_info_0.1_0.7"
OUT_LIST = "data/megadepth/index/trainvaltest_list/train_list_subset20.txt"

LOW_QUALITY = {'0000', '0002', '0011', '0020', '0033', '0050', '0103', '0105',
               '0143', '0176', '0177', '0265', '0366', '0474', '0860', '4541'}

with open(LIST_PATH) as f:
    lines = [l.strip() for l in f if l.strip()]

scored = []
for l in lines:
    scene_id = l.split('.')[0].split('_')[0]
    if scene_id in LOW_QUALITY:
        continue
    npz_file = l if l.endswith('.npz') else f"{l}.npz"
    npz = np.load(f"{NPZ_ROOT}/{npz_file}", allow_pickle=True)
    scored.append((len(npz['pair_infos']), scene_id, l))

scored.sort(reverse=True)  # most pairs first

chosen_ids = []
for n_pairs, scene_id, l in scored:
    if scene_id in chosen_ids:
        continue
    chosen_ids.append(scene_id)
    if len(chosen_ids) == N:
        break

# grab ALL overlap-bin files for each chosen scene (free extra pairs, same depth/images already covers all bins)
keep = [l for l in lines if l.split('.')[0].split('_')[0] in chosen_ids]

with open(OUT_LIST, 'w') as f:
    f.write('\n'.join(keep) + '\n')

with open("data/megadepth/index/scene_ids_subset20.txt", 'w') as f:
    f.write('\n'.join(chosen_ids) + '\n')

print(f"Picked the {N} richest scenes -> {OUT_LIST}\n")
for n_pairs, scene_id, l in scored:
    if l in keep:
        print(f"  {scene_id}  ({n_pairs} pairs)")

print("\nScene IDs:", ' '.join(chosen_ids), "\n")

patterns = ' '.join(f"--wildcards 'phoenix/S6/zoom_1/{s}/*'" for s in chosen_ids)
print("# 1. download depths for exactly these scenes (streams 199GB, keeps only matches):")
print("mkdir -p data/megadepth/train")
print(f"curl -L http://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz "
      f"| tar -xz {patterns} -C data/megadepth/train")
print()
print("# 2. then in the D2-Net Drive folder (Undistorted_SfM/), ctrl-click + download just these IDs:")
print(' '.join(chosen_ids))