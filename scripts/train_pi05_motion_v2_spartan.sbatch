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
#SBATCH --job-name=pi05_motion_v2_spartan_balls_shared

# pi05_motion_v2 on spartan_balls. Same hyperparams as the spartan motion v1
# shared-obs run (eff batch 8*2*8=128, 10k steps) plus the V2 conditioning
# (4096-d SAM2 latent + 10-vector action history -> VLM prefix + adaRMS).

set -euo pipefail
echo "=== $(date -Iseconds)  job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURMD_NODENAME ==="
nvidia-smi -L

source /home/dfeng8/miniconda3/etc/profile.d/conda.sh
conda activate vlash

PROJECT=/srv/disk00/dfeng8/work/dynamics/vlash
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vlash.cli train examples/train/pi05_motion_v2/spartan_balls_motion_v2.yaml

echo "=== $(date -Iseconds)  vlash exited $? ==="
