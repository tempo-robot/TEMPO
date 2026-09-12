# Closing the Representational Gap for VLAs in Dynamic Settings

**Temporal Encoding for Motion-aware Policy (TEMPO)**

<!-- TODO: affiliation --> _Affiliation_

<!-- TODO: author list with links --> _Author One, Author Two, Author Three_

[[arXiv]](#-citation) [[Project Page]](#-citation) [[BibTeX]](#-citation)

<!-- TODO: add the method figure at assets/main.png and uncomment
![main figure](assets/main.png)
-->

A single-frame vision-language-action policy `π(o_t)` sees **where** things are but not **where
they are going**, and it cannot tell apart visually identical frames that require different
actions. TEMPO closes both gaps with two cheap temporal channels on a pretrained π0.5:

- **TEMPO<sub>MOT</sub> — scene motion.** Causal temporal attention across the K-frame
  observation history, applied at every 4th SigLIP layer and reusing that layer's Q/K/V (no new
  parameters, backbone token count unchanged); plus a frozen SAM2 memory-attention feature of
  the head camera, fused into the current head-cam tokens by a zero-gated cross-attention block.
- **TEMPO<sub>ACT</sub> — proprioceptive history.** Ten bucket-mean past-action vectors over a
  10 s window, injected as extra VLM prefix tokens and as a zero-init residual on the action
  expert's adaRMS conditioning.

**Two prediction heads.** By default the policy flow-matches a compact latent
**u<sub>t</sub>** — the MVAE code μ<sub>v</sub> of the observation window starting at *t* — and a
frozen decoder reads the action chunk out of it. `predict_ut=False` switches back to standard
action-chunk regression.

## 🗓️ TODO

**Done**

- [x] **TEMPO<sub>MOT</sub>** — space-time separable attention over the observation history, and
      the frozen SAM2 motion cue fused into the head-cam tokens
- [x] **TEMPO<sub>ACT</sub>** — bucket-mean action history as prefix tokens plus a zero-init
      adaRMS residual
- [x] Swappable prediction head: u<sub>t</sub> latent (default) or action-chunk regression
- [x] u<sub>t</sub> pipeline: PoE MVAE → μ<sub>v</sub> encoder → decoder trainer
- [x] TEMPO<sub>MOT</sub> SAM2 token precompute
- [x] Training: multi-GPU DDP, warm start from π0.5
- [x] Inference on a trained checkpoint: policy server and client

**To do**

- [ ] Release the training datasets (TEMPO-Bench)
- [ ] Release trained checkpoints
- [ ] Deployment sample code for the I2RT YAM arm — the online TEMPO<sub>MOT</sub> /
      TEMPO<sub>ACT</sub> buffers a live control loop needs (K-frame image ring buffer,
      streaming SAM2 tokens, executed-action history)
- [ ] arXiv preprint and project page

## 🛠️ Installation

Requires Python 3.11+, an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/). CUDA libraries come
in through uv — no system CUDA install needed.

```bash
git clone <this repo> && cd TEMPO
uv sync
```

On **Blackwell GPUs** (sm_120: RTX PRO 6000, RTX 50-series, B200) the pinned `torch==2.7.1`
resolves to a `+cu126` wheel whose kernels stop at `sm_90`. Install the cu128 build over it:

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Patch `transformers` — TEMPO<sub>MOT</sub>'s space-time attention lives in a modified SigLIP, and
the model will not build without it. Use `-f`, since a shell with `cp` aliased to `cp -i` skips
the overwrites silently:

```bash
cp -rf src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

Re-run that copy after editing anything under `transformers_replace/` — the model imports the
*installed* `transformers`, not the source tree.

Set up SAM2 for the TEMPO<sub>MOT</sub> motion cue (its dependencies are not in the lockfile):

```bash
git clone https://github.com/facebookresearch/sam2.git
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
uv pip install "hydra-core>=1.3.2" "iopath>=0.1.10"
```

Environment:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot/datasets
export LEROBOT_VIDEO_BACKEND=pyav
export SAM2_REPO=/path/to/sam2
export SAM2_CKPT=/path/to/checkpoints/sam2.1_hiera_tiny.pt

# optional: move the caches and checkpoints off the repo, e.g. onto a scratch disk
# export TEMPO_DATA_ROOT=/scratch/$USER/tempo/caches
# export TEMPO_CKPT_ROOT=/scratch/$USER/tempo/checkpoints
```

The configs expect their artifacts at these paths, all relative to the working directory unless
you set the two variables above:

```
caches/sam2/<repo_id>/            SAM2 token cache        (step 1)
caches/ut/<repo_id>/              u_t window sidecars     (step 3b)
caches/ut_decoder/<repo_id>/      u_t decoder             (step 3c)
checkpoints/pi05_base_pytorch/    converted pi0.5 weights (below)
checkpoints/<config>/<exp>/<step> your own runs, for warm starts
```

Run the commands below from the repo root and everything lands where the configs look for it.

Verify:

```bash
uv run python -c "
import torch, inspect, transformers.models.siglip.modeling_siglip as m
import openpi.training.config as C
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
print('torch', torch.__version__, '| arch', torch.cuda.get_arch_list()[-1])
assert '_temporal_attention' in inspect.getsource(m), 'transformers_replace not installed'
c = C.get_config('pi05_yam_tempo_ut_dynamic_handover')
print('config OK | predict_ut =', c.model.predict_ut, '| flow_dim =', c.model.flow_dim)
PI0Pytorch(c.model); print('model builds OK')
"
```

Finally, get a PyTorch π0.5 base checkpoint to fine-tune from:

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

Set `OPENPI_DATA_HOME` first if you want the ~12 GB download somewhere other than `~/.cache`.

## 🧑‍💻 Usage

### Config matrix

Every channel is a separate config, so each can be ablated independently:

| config | MOT visual memory | MOT SAM2 cue | ACT history | head |
|---|---|---|---|---|
| `pi05_yam_*` | — | — | — | action chunk |
| `pi05_yam_tempo_mot_video_*` | ✓ (6 frames) | — | — | action chunk |
| `pi05_yam_tempo_mot_*` | ✓ (6 frames) | ✓ | — | action chunk |
| `pi05_yam_tempo_*` | ✓ (3 frames) | ✓ | ✓ | action chunk |
| `pi05_yam_tempo_ut_*` | ✓ (3 frames) | ✓ | ✓ | **u<sub>t</sub>** |

All defined in `src/openpi/training/config.py`, with their cache and checkpoint paths derived
from `TEMPO_DATA_ROOT` / `TEMPO_CKPT_ROOT` (see Installation), so they work from a fresh clone.
The knobs:

| key | meaning |
|---|---|
| `obs_history`, `obs_history_stride_s` | memory frames K (incl. current) and their spacing; `1` == plain single-frame π0.5 |
| `temporal_attn_period` | apply temporal attention every N-th ViT layer |
| `use_sam2_fusion`, `sam2_num_tokens`, `sam2_token_dim` | the SAM2 motion cue |
| `use_action_history`, `action_history_steps`, `action_history_dim` | TEMPO<sub>ACT</sub> (`_dim` = raw action dim: 14 dual-arm, 7 single-arm) |
| `predict_ut` | the head switch: u<sub>t</sub> latent (default) or action-chunk regression |
| `ut_dim`, `ut_window`, `ut_decoder_path` | latent width, decoded chunk length (must equal `action_horizon`), frozen decoder |

### 1. Precompute the TEMPO<sub>MOT</sub> SAM2 tokens

SAM2 is frozen, so its features are extracted once per dataset rather than in the training loop:

```bash
uv run python tools/precompute_sam2_tokens.py \
    --root  $HF_LEROBOT_HOME/<repo_id> \
    --cache caches/sam2/<repo_id> \
    --start 0 --end 90
```

Each episode becomes `episode_NNNNNN.npz` holding `latent (T, 256, 8, 8)` — the
post-memory-attention feature of the head camera — plus the raw `action` and `state` columns.
TEMPO<sub>ACT</sub> derives its history from `action` in the same file, so
`action_history_cache_dir` is normally the same path and needs no second pass.

### 2. Norm stats

```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name pi05_yam_tempo_ut_dynamic_handover
```

### 3. Build the u<sub>t</sub> artifacts

Skip this if you are training with `predict_ut=False`. Three stages, all reading the SAM2 cache:

```bash
# (a) Product-of-Experts MVAE over (SAM2 window, action chunk)
uv run python tools/ut/train_mvae.py \
    --cache caches/sam2/<repo_id> --out caches/mvae/<repo_id> \
    --latent-dim 64 --beta 0 --pool-spatial

# (b) per-episode sidecars: mu_v for the window starting at each frame
uv run python tools/ut/encode_ut_windows.py \
    --mvae caches/mvae/<repo_id>/checkpoints/last.pt \
    --cache caches/sam2/<repo_id> --out caches/ut/<repo_id>

# (c) the decoder the policy uses at inference: mu_v -> action chunk
uv run python tools/ut/train_ut_decoder.py \
    --mvae-ckpt caches/mvae/<repo_id>/checkpoints/last.pt \
    --sam-cache caches/sam2/<repo_id> --ut-dir caches/ut/<repo_id> \
    --norm-stats assets/<config_name>/<repo_id> \
    --out caches/ut_decoder/<repo_id>
```

`--norm-stats` is required: the decoder is trained in the policy's normalized action space so
its output can be returned from `sample_actions` unchanged.

If you omit `--split`, stages (a) and (b) derive a deterministic 80/20 episode split and save it
next to the run; pass that `split.json` to later stages to reuse it.

### 4a. Train u<sub>t</sub>-based TEMPO (default)

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_yam_tempo_ut_dynamic_handover \
    --exp_name=my_run --batch_size=128 --checkpoint_base_dir=./checkpoints
```

`batch_size` is the global batch. Checkpoints land in
`<checkpoint_base_dir>/<config_name>/<exp_name>/<step>/`.

### 4b. Train action-chunk-based TEMPO

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_yam_tempo_dynamic_handover \
    --exp_name=my_run --batch_size=128 --checkpoint_base_dir=./checkpoints
```

To flip a u<sub>t</sub> config over instead, one flag is enough — `ut_cache_dir` and
`ut_decoder_path` are then ignored:

```bash
uv run torchrun ... scripts/train_pytorch.py pi05_yam_tempo_ut_dynamic_handover \
    --exp_name=chunk_head --model.no-predict-ut
```

tyro renders booleans as a `--flag` / `--no-flag` pair, so use `--model.no-predict-ut`;
`--model.predict_ut=False` is rejected.

Both heads fine-tune from a `pi05_yam_tempo_mot_video_*` checkpoint — train that first, with
`--exp_name=my_run`, or point `pytorch_weight_path` at `checkpoints/pi05_base_pytorch` to start
from π0.5 directly. The training script re-initializes tensors whose shape
changed, so the backbone, TEMPO<sub>MOT</sub> and TEMPO<sub>ACT</sub> weights transfer when you
swap heads.

### 5. Inference with a trained checkpoint

Serve a checkpoint over websockets:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_yam_tempo_ut_dynamic_handover \
    --policy.dir=checkpoints/pi05_yam_tempo_ut_dynamic_handover/my_run/5000 \
    --port=8000
```

and query it for an action chunk:

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
action_chunk = client.infer({
    "state": state,                       # (A,) float32
    "images": {                           # HWC uint8; "head" is required
        "head": head_rgb,
        "left_wrist": left_rgb,
        "right_wrist": right_rgb,
    },
    "prompt": "dynamic handover",
    "sam2_tokens": sam2_tokens,           # (64, 256) float32, TEMPO-MOT
    "action_history": action_history,     # (10, A) float32 bucket means, TEMPO-ACT
})["actions"]                             # (action_horizon, A) raw units
```

The server auto-detects the PyTorch checkpoint and loads the matching config. The
u<sub>t</sub> head needs nothing extra from the caller — u<sub>t</sub> is what the policy
predicts, and the frozen decoder turns it into the action chunk server-side.

To run in-process instead of over a socket, `openpi.policies.policy_config.create_trained_policy`
takes the same config and checkpoint directory and exposes the same `.infer()`. See
`docs/remote_inference.md`, `examples/simple_client/`, and `docs/docker.md`.

## 📊 Dataset

TEMPO trains on [LeRobot](https://github.com/huggingface/lerobot) datasets (v2.1 or v3.0) placed
under `$HF_LEROBOT_HOME/<repo_id>`:

```
<repo_id>/
├── meta/
│   ├── info.json          # codebase_version, robot_type, fps, total_episodes, features
│   ├── episodes.jsonl     # per-episode length and task
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl        # task strings, used as the prompt when prompt_from_task=True
├── data/chunk-000/
│   └── episode_NNNNNN.parquet
└── videos/chunk-000/
    ├── observation.images.head/episode_NNNNNN.mp4
    ├── observation.images.left_wrist/episode_NNNNNN.mp4
    └── observation.images.right_wrist/episode_NNNNNN.mp4
```

Each parquet holds one episode, one row per frame:

| column | shape | notes |
|---|---|---|
| `action` | (A,) float32 | A = 14 dual-arm, 7 single-arm |
| `observation.state` | (A,) float32 | joint positions + gripper |
| `frame_index` | int | 0-based within the episode |
| `episode_index` | int | matches the `NNNNNN` in the file names |
| `timestamp`, `index`, `task_index` | | standard LeRobot columns |

Requirements:

- A **`head` camera is mandatory**. `left_wrist` and `right_wrist` are optional and are
  zero-filled with `image_mask=False` when absent; declare which exist via `cameras` on the data
  config.
- **fps must match `action_history_fps`** (30 by default) — TEMPO<sub>ACT</sub> uses it to
  convert its window from seconds to frames.
- Set `action_dim` on the data config and `action_history_dim` on the model config to A.
- Video frame count must equal the parquet row count and the length recorded in
  `meta/episodes.jsonl`; the SAM2 precompute asserts this.
- The SAM2, action-history and u<sub>t</sub> caches are keyed by `episode_index` and
  `frame_index`, so episode numbering must stay consistent between the dataset and the caches.

## ❤️ Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (π0 / π0.5) and
[LeRobot](https://github.com/huggingface/lerobot). The TEMPO<sub>MOT</sub> motion cue uses
[SAM 2](https://github.com/facebookresearch/sam2); the visual-memory encoder follows the
space-time separable attention of Torne et al., *Multi-Scale Embodied Memory for Vision Language
Action Models* ([arXiv:2603.03596](https://arxiv.org/abs/2603.03596)). π0.5 weights are
distributed by Physical Intelligence; Gemma is used under the license in `LICENSE_GEMMA.txt`.

## 📝 Citation

<!-- TODO: replace with the published entry -->

```bibtex
@misc{tempo2026closing,
      title={Closing the Representational Gap for VLAs in Dynamic Settings},
      author={TODO},
      year={2026},
      eprint={TODO},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={TODO},
}
```
