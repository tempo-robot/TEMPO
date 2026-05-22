#!/bin/bash
#SBATCH --account=unveil.a
#SBATCH --partition=unveil.p
#SBATCH --nodelist=denali-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/home/dfeng8/sam2_uvt/logs/%x_%j.out
#SBATCH --error=/home/dfeng8/sam2_uvt/logs/%x_%j.err
#SBATCH --job-name=pi05_motion_dynamic_pour_right

# Pipeline: SAM2 extract (head cam, dynamic_pour_right) -> motion sidecar -> vlash train.
# Idempotent.

set -euo pipefail
echo "=== $(date -Iseconds)  job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURMD_NODENAME ==="
nvidia-smi -L

source /home/dfeng8/miniconda3/etc/profile.d/conda.sh
conda activate vlash

PROJECT=/srv/disk00/dfeng8/work/dynamics/vlash
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAM_CACHE=/scratch/dfeng8/sam2_uvt_cache/sam2.1_t_head_g8_dynamic_pour_right
MOTION_DIR=/scratch/dfeng8/sam2_uvt_cache/motion_diff_first_last_dynamic_pour_right
DATA_ROOT=/srv/disk00/dfeng8/work/dynamics/data/dynamic_pour_right

N_EP=$(ls $DATA_ROOT/data/chunk-000/episode_*.parquet 2>/dev/null | wc -l)
mkdir -p "$SAM_CACHE"
NEED_SAM=$(python -c "
from pathlib import Path
d = Path('$SAM_CACHE'); n = len(list(d.glob('episode_*.npz')))
print(0 if n >= $N_EP else 1)
")
if [ "$NEED_SAM" = "1" ]; then
    echo "=== Step 1: extract SAM2 latents -> $SAM_CACHE ==="
    CUDA_VISIBLE_DEVICES=0 python /home/dfeng8/sam2_uvt/precompute_latents.py \
        --root "$DATA_ROOT" --cache "$SAM_CACHE" --start 0 --end "$N_EP"
else
    echo "=== Step 1: SAM2 latents already cached, skipping ==="
fi

NEED_MOTION=$(python -c "
from pathlib import Path
d = Path('$MOTION_DIR'); n = len(list(d.glob('episode_*.npz')))
print(0 if n >= $N_EP else 1)
")
if [ "$NEED_MOTION" = "1" ]; then
    echo "=== Step 2: motion sidecars -> $MOTION_DIR ==="
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
        if t - W >= 0:
            mp[t] = sam[t - 1] - sam[t - W]; pp[t] = False
        if t + W <= T:
            mf[t] = sam[t + W - 1] - sam[t]; fp[t] = False
    np.savez_compressed(out_dir / f.name, motion_past=mp, motion_future=mf,
                         motion_past_is_pad=pp, motion_future_is_pad=fp, ep_len=np.int32(T))
print('done')
"
else
    echo "=== Step 2: motion sidecars already present, skipping ==="
fi

echo "=== Step 3: launch vlash train ==="
python -m vlash.cli train examples/train/pi05_motion/dynamic_pour_right_motion.yaml

echo "=== $(date -Iseconds)  vlash exited $? ==="
