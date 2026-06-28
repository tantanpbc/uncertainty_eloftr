#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."   # repo root

echo "== Step 1/4: picking 20 scenes from your index =="
python scripts/pick_and_setup_subset.py
SCENES=$(cat data/megadepth/index/scene_ids_subset20.txt)
echo ""
echo "Chosen scene IDs: $SCENES"
echo ""

echo "== Step 2/4: downloading depth maps for exactly these scenes =="
echo "(streams the 199GB Cornell tar, keeps only matching scenes on disk)"
mkdir -p data/megadepth/train
PATTERNS=""
for s in $SCENES; do PATTERNS="$PATTERNS --wildcards phoenix/S6/zoom_1/$s/*"; done
curl -L http://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz \
  | tar -xz $PATTERNS -C data/megadepth/train
echo "Depths done."
echo ""

echo "== Step 3/4: manual step — undistorted images =="
echo "Open this in your browser:"
echo "  https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf"
echo "Go into Undistorted_SfM/, ctrl-click exactly these folders, then click Download:"
echo "  $SCENES"
echo ""
echo "Once downloaded and unzipped, move it into place with:"
echo "  mv ~/Downloads/Undistorted_SfM data/megadepth/train/Undistorted_SfM"
echo "  (or merge if the folder already exists: rsync -av ~/Downloads/Undistorted_SfM/ data/megadepth/train/Undistorted_SfM/)"
read -p "Press Enter once that's done and moved into place to continue verification... "
echo ""

echo "== Step 4/4: verifying + one real load test =="
ALL_OK=1
for s in $SCENES; do
  if [ -d "data/megadepth/train/phoenix/S6/zoom_1/$s/depths" ] && [ -d "data/megadepth/train/Undistorted_SfM/$s/images" ]; then
    echo "  $s: OK"
  else
    echo "  $s: MISSING (check depths/images dirs above)"
    ALL_OK=0
  fi
done

if [ "$ALL_OK" -eq 1 ]; then
  FIRST_NPZ=$(head -1 data/megadepth/index/trainvaltest_list/train_list_subset20.txt)
  case "$FIRST_NPZ" in *.npz) ;; *) FIRST_NPZ="${FIRST_NPZ}.npz" ;; esac
  echo ""
  echo "All 20 present. Running one real load test on $FIRST_NPZ ..."
  python -c "
from src.datasets.megadepth import MegaDepthDataset
ds = MegaDepthDataset('data/megadepth/train',
                       'data/megadepth/index/scene_info_0.1_0.7/$FIRST_NPZ',
                       mode='train', min_overlap_score=0.0, img_resize=832, df=8,
                       img_padding=True, depth_padding=True)
d = ds[0]
print('OK ->', d['image0'].shape, d['depth0'].shape)
"
  echo ""
  echo "Ready. Train with:"
  echo "  python -u train.py configs/data/megadepth_subset20.py configs/loftr/eloftr_finetune_uncertainty.py \\"
  echo "      --exp_name uncertainty_ft --ckpt_path weights/eloftr_full.ckpt \\"
  echo "      --gpus 1 --batch_size 4 --max_epochs 3 --thr 0.1"
else
  echo ""
  echo "Some scenes missing — fix step 2 or 3 above before training."
fi