# Closing the Representational Gap for VLAs in Dynamic Settings

**Temporal Encoding for Motion-aware Policy (TEMPO)**

<!-- TODO: affiliation --> _Affiliation_

<!-- TODO: author list with links --> _Author One, Author Two, Author Three_

[[arXiv]](#-citation) [[Project Page]](#-citation) [[BibTeX]](#-citation)

<!-- TODO: add the method figure at assets/main.png and uncomment
![main figure](assets/main.png)
-->

This repository contains the official implementation of Temporal Encoding for Motion-aware
Policy (TEMPO) from the paper "Closing the Representational Gap for VLAs in Dynamic Settings,"
accepted at CoRL 2026. TEMPO gives a single-frame VLA the temporal context it lacks through two
lightweight channels, scene motion (TEMPO<sub>MOT</sub>) and proprioceptive history
(TEMPO<sub>ACT</sub>), adding about 2M parameters (0.08%) without altering the pretrained
backbone. By default the policy predicts a compact latent u<sub>t</sub> that a frozen decoder
turns into the action chunk; `predict_ut=False` restores standard action-chunk regression.

## 🗓️ TODO

**Done**

- [x] TEMPO<sub>MOT</sub>
- [x] TEMPO<sub>ACT</sub>
- [x] u<sub>t</sub> prediction head
- [x] u<sub>t</sub> MVAE pipeline
- [x] SAM2 token precompute
- [x] Training code
- [x] Inference code

**To do**

- [ ] Release training data
- [ ] Release trained checkpoints
- [ ] I2RT YAM deployment
- [ ] arXiv and project page

## 🛠️ Installation

Requires Python 3.11+, an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd TEMPO
uv sync

# Blackwell GPUs (sm_120) only: the pinned torch resolves to a cu126 wheel with no sm_120 kernels
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# required: TEMPO-MOT's attention lives in a patched SigLIP, and the model will not build without it
cp -rf src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# SAM2, for the TEMPO-MOT motion cue
git clone https://github.com/facebookresearch/sam2.git
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
uv pip install "hydra-core>=1.3.2" "iopath>=0.1.10"
```

Use `cp -rf`: a shell with `cp` aliased to `cp -i` skips the overwrites silently. Re-run it after
editing anything under `transformers_replace/`, since the model imports the installed
`transformers`, not the source tree.

```bash
export HF_LEROBOT_HOME=/path/to/lerobot/datasets
export LEROBOT_VIDEO_BACKEND=pyav
export SAM2_REPO=/path/to/sam2
export SAM2_CKPT=/path/to/checkpoints/sam2.1_hiera_tiny.pt

# optional: put the caches and checkpoints somewhere other than the repo
# export TEMPO_DATA_ROOT=/scratch/$USER/tempo/caches
# export TEMPO_CKPT_ROOT=/scratch/$USER/tempo/checkpoints
```

Verify the install:

```bash
uv run python -c "
import torch, inspect, transformers.models.siglip.modeling_siglip as m
import openpi.training.config as C
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
print('torch', torch.__version__, '| arch', torch.cuda.get_arch_list()[-1])
assert '_temporal_attention' in inspect.getsource(m), 'transformers_replace not installed'
PI0Pytorch(C.get_config('pi05_yam_tempo_ut_dynamic_handover').model); print('OK')
"
```

Get a PyTorch π0.5 checkpoint to fine-tune from:

```bash
uv run python -c "
import openpi.shared.download as d
print(d.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))
"
uv run python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
    --config_name pi05_yam_dynamic_handover \
    --output_path checkpoints/pi05_base_pytorch
```

## 🧑‍💻 Usage

Run everything from the repo root. Each config reads its artifacts from paths under
`TEMPO_DATA_ROOT` / `TEMPO_CKPT_ROOT`, so the commands below write exactly where they are read.

| config | TEMPO<sub>MOT</sub> | TEMPO<sub>ACT</sub> | head |
|---|---|---|---|
| `pi05_yam_*` | - | - | action chunk |
| `pi05_yam_tempo_mot_*` | ✓ | - | action chunk |
| `pi05_yam_tempo_*` | ✓ | ✓ | action chunk |
| `pi05_yam_tempo_ut_*` | ✓ | ✓ | **u<sub>t</sub>** |

Each row adds one component. Defined in `src/openpi/training/config.py`; the knobs are
`obs_history` / `obs_history_stride_s` / `temporal_attn_period` / `use_sam2_fusion`
(TEMPO<sub>MOT</sub>), `use_action_history` / `action_history_*` (TEMPO<sub>ACT</sub>), and
`predict_ut` / `ut_dim` / `ut_decoder_path` (the head).

### 1. SAM2 tokens

SAM2 is frozen, so its features are extracted once per dataset:

```bash
uv run python tools/precompute_sam2_tokens.py \
    --root $HF_LEROBOT_HOME/<repo_id> --cache caches/sam2/<repo_id> --start 0 --end 90
```

Each episode becomes `episode_NNNNNN.npz` with `latent (T, 256, 8, 8)` plus the raw `action` and
`state` columns. TEMPO<sub>ACT</sub> reads its history from `action` in the same file.

### 2. Norm stats

```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name pi05_yam_tempo_ut_dynamic_handover
```

### 3. u_t artifacts

Skip if training with `predict_ut=False`.

```bash
R=<repo_id>
uv run python tools/ut/train_mvae.py --cache caches/sam2/$R --out caches/mvae/$R \
    --latent-dim 64 --beta 0 --pool-spatial

uv run python tools/ut/encode_ut_windows.py --mvae caches/mvae/$R/checkpoints/last.pt \
    --cache caches/sam2/$R --out caches/ut/$R

uv run python tools/ut/train_ut_decoder.py --mvae-ckpt caches/mvae/$R/checkpoints/last.pt \
    --sam-cache caches/sam2/$R --ut-dir caches/ut/$R \
    --norm-stats assets/<config_name>/$R --out caches/ut_decoder/$R
```

`--norm-stats` is required: the decoder is trained in the policy's normalized action space, so
its output can be returned from `sample_actions` unchanged. Omit `--split` and the first two
stages derive a deterministic 80/20 episode split and save it next to the run.

### 4. Train

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_yam_tempo_ut_dynamic_handover \
    --exp_name=my_run --batch_size=128 --checkpoint_base_dir=./checkpoints
```

`batch_size` is the global batch. Checkpoints land in
`<checkpoint_base_dir>/<config_name>/<exp_name>/<step>/`. Swap `pi05_yam_tempo_ut_*` for
`pi05_yam_tempo_*` to regress action chunks, or flip an existing u<sub>t</sub> config with
`--model.no-predict-ut` (tyro renders booleans as a `--flag` / `--no-flag` pair, so
`--model.predict_ut=False` is rejected). Configs fine-tune from the rung below them, or from
`checkpoints/pi05_base_pytorch` to start from π0.5 directly; tensors whose shape changed are
re-initialized, so the rest transfers when you swap heads.

### 5. Inference

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_yam_tempo_ut_dynamic_handover \
    --policy.dir=checkpoints/pi05_yam_tempo_ut_dynamic_handover/my_run/5000 --port=8000
```

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
action_chunk = client.infer({
    "state": state,                       # (A,) float32
    "images": {"head": head_rgb, "left_wrist": left_rgb, "right_wrist": right_rgb},  # HWC uint8
    "prompt": "dynamic handover",
    "sam2_tokens": sam2_tokens,           # (64, 256) float32, TEMPO-MOT
    "action_history": action_history,     # (10, A) float32 bucket means, TEMPO-ACT
})["actions"]                             # (action_horizon, A) raw units
```

The u<sub>t</sub> head needs nothing extra from the caller: the frozen decoder turns the
predicted latent into the action chunk server-side. For in-process use,
`openpi.policies.policy_config.create_trained_policy` exposes the same `.infer()`.

## 📊 Dataset

[LeRobot](https://github.com/huggingface/lerobot) datasets (v2.1 or v3.0) under
`$HF_LEROBOT_HOME/<repo_id>`:

```
<repo_id>/
├── meta/         info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl
├── data/chunk-000/episode_NNNNNN.parquet
└── videos/chunk-000/observation.images.{head,left_wrist,right_wrist}/episode_NNNNNN.mp4
```

Each parquet holds one episode, one row per frame, with `action` (A,) and `observation.state`
(A,) float32, plus the standard `frame_index` / `episode_index` / `timestamp` / `index` /
`task_index` columns. A = 14 for a dual-arm robot, 7 for single-arm.

- A `head` camera is required; the wrist views are optional and zero-filled when absent. Declare
  which exist via `cameras` on the data config.
- fps must match `action_history_fps` (30 by default), which converts the TEMPO<sub>ACT</sub>
  window from seconds to frames.
- Set `action_dim` on the data config and `action_history_dim` on the model config to A.
- Video frame count must equal the parquet row count and the length in `meta/episodes.jsonl`.
- The caches are keyed by `episode_index` and `frame_index`, so episode numbering must stay
  consistent between the dataset and the caches.

## ❤️ Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (π0 / π0.5) and
[LeRobot](https://github.com/huggingface/lerobot). TEMPO<sub>MOT</sub> uses
[SAM 2](https://github.com/facebookresearch/sam2). π0.5 weights are distributed by Physical
Intelligence; Gemma is used under the license in `LICENSE_GEMMA.txt`.

## 📝 Citation

<!-- TODO: replace with the published entry -->

```bibtex
@inproceedings{tempo2026closing,
      title={Closing the Representational Gap for VLAs in Dynamic Settings},
      author={TODO},
      booktitle={Conference on Robot Learning (CoRL)},
      year={2026},
      url={TODO},
}
```
