#!/bin/bash
# Run on denali-15 outside SLURM, using GPUs 2-7 (avoiding the 2 non-slurm
# processes on GPUs 0-1). Effective batch = bs=11 * grad_accum=2 * 6 GPUs = 132.

set -euo pipefail
echo "=== $(date -Iseconds)  on $(hostname) ==="

source /home/dfeng8/miniconda3/etc/profile.d/conda.sh
conda activate vlash

PROJECT=/srv/disk00/dfeng8/work/dynamics/vlash  # denali-15 vlash editable install
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

SAM_CACHE=/scratch/dfeng8/sam2_uvt_cache/sam2.1_t_head_g8_dynamic_pour_right_despiked
MOTION_DIR=/scratch/dfeng8/sam2_uvt_cache/motion_diff_first_last_dynamic_pour_right_despiked
DATA_ROOT=/srv/disk00/dfeng8/work/dynamics/data/dynamic_pour_right_despiked

# Stage data on denali-15 if not present.
if [ ! -d "$DATA_ROOT" ]; then
    echo "rsyncing despiked pour dataset from denali-14 ..."
    mkdir -p "$DATA_ROOT"
    rsync -a denali-14:$DATA_ROOT/ "$DATA_ROOT/"
fi

# --- Step 1: SAM2 extract (single GPU = first of the 6 visible) ---
N_EP=$(ls $DATA_ROOT/data/chunk-000/episode_*.parquet 2>/dev/null | wc -l)
mkdir -p "$SAM_CACHE"
n_sam=$(ls $SAM_CACHE 2>/dev/null | grep -c "episode_" || true)
if [ "$n_sam" -lt "$N_EP" ]; then
    echo "=== Step 1: extract SAM2 latents ($N_EP eps) ==="
    CUDA_VISIBLE_DEVICES=2 python /home/dfeng8/sam2_uvt/precompute_latents.py \
        --root "$DATA_ROOT" --cache "$SAM_CACHE" --start 0 --end "$N_EP"
else
    echo "=== Step 1: SAM2 latents already cached ($n_sam) ==="
fi

# --- Step 2: motion sidecars (CPU) ---
mkdir -p "$MOTION_DIR"
n_mo=$(ls $MOTION_DIR 2>/dev/null | grep -c "episode_" || true)
if [ "$n_mo" -lt "$N_EP" ]; then
    echo "=== Step 2: motion sidecars ==="
    python -c "
import numpy as np
from pathlib import Path
W = 16; D = 256
sam_dir = Path('$SAM_CACHE'); out_dir = Path('$MOTION_DIR'); out_dir.mkdir(parents=True, exist_ok=True)
for f in sorted(sam_dir.glob('episode_*.npz')):
    with np.load(f) as z: lat = z['latent']
    sam = lat.mean(axis=(2, 3)); T = sam.shape[0]
    mp = np.zeros((T, D), dtype=np.float32); mf = np.zeros_like(mp)
    pp = np.ones(T, dtype=bool); fp = np.ones(T, dtype=bool)
    for t in range(T):
        if t - W >= 0: mp[t] = sam[t - 1] - sam[t - W]; pp[t] = False
        if t + W <= T: mf[t] = sam[t + W - 1] - sam[t]; fp[t] = False
    np.savez_compressed(out_dir / f.name, motion_past=mp, motion_future=mf,
                         motion_past_is_pad=pp, motion_future_is_pad=fp, ep_len=np.int32(T))
print('done')
"
else
    echo "=== Step 2: motion sidecars already present ($n_mo) ==="
fi

# --- Step 3: train on 6 GPUs (denali-15 GPUs 2-7), effective bs = 11*2*6 = 132 ---
echo "=== Step 3: launch 6-GPU vlash train ==="
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# Vlash's CLI auto-detects 6 GPUs from CUDA_VISIBLE_DEVICES and invokes
# accelerate launch --multi_gpu --num_processes=6.
python -m vlash.cli train examples/train/pi05_motion/dynamic_pour_right_despiked_motion.yaml

echo "=== $(date -Iseconds)  done ==="
