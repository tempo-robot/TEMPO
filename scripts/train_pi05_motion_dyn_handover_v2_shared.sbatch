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
#SBATCH --job-name=pi05_motion_dyn_handover_v2_shared

# pi05_motion on dynamic_handover_v2 with shared_observation=true and
# max_delay_steps=8 to match baseline training. 8 GPUs × bs=8 × ga=2 = 128.

set -euo pipefail
echo "=== $(date -Iseconds)  job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURMD_NODENAME ==="
nvidia-smi -L

source /home/dfeng8/miniconda3/etc/profile.d/conda.sh
conda activate vlash

PROJECT=/srv/disk00/dfeng8/work/dynamics/vlash
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vlash.cli train examples/train/pi05_motion/dynamic_handover_v2_3cam_motion.yaml

echo "=== $(date -Iseconds)  vlash exited $? ==="
