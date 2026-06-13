# lerobot-rlinf

**SO-101 + π₀.₅ post-training glue, currently in the SFT phase.**
Drives `openpi` LoRA training and `leisaac` Isaac Lab eval as two cooperating processes,
plus a dataset-only diagnostics framework for closing the loop on training quality.

> 中文版见 [README.zh.md](README.zh.md)

---

## What this repo actually is

Three small Python subsystems on top of three vendored submodules — the submodules do
the heavy lifting (training kernel, simulator, robot model), this repo wires them
together for the SO-ARM100 / SO-101 arm and adds the analysis layer on top.

```
                       ┌─────────────────────────────────────────┐
   sft_train.py  ───►  │  third_party/openpi  (LoRA π₀.₅ kernel) │
                       └─────────────────────────────────────────┘
                                       │ SFT checkpoints/
                                       ▼
   eval.py ──► spawns ──► openpi serve_policy.py  (WebSocket)
            └► spawns ──► eval_run.py ──► third_party/leisaac (Isaac Lab env)
                                                       │
                                                       ▼
                            lerobot v2.1 dataset (data/ + videos/ + meta/)
                                                       │
                                                       ▼
                          python -m src.diagnostics  (ref vs candidate)
                                                       │
                                                       ▼
                                  diagnostics report (.json + .md)
                                                       │
                                          findings drive reward shaping
                                                       ▼
   src/rl/simple/train.py  ──► frozen π₀.₅ (serve_policy) + Gaussian residual head
                                                       │  PPO + EXP_03/05-aware
                                                       │  reward (OOD penalty,
                                                       │  survival cost, dense lift)
                                                       ▼
                              residual head ckpt  ──► src/rl/simple/eval.py
```

What's **live** today:
- SFT (supervised fine-tuning) of LoRA π₀.₅ on the LightwheelAI/leisaac-pick-orange dataset
- Headless Isaac Lab eval that writes one mp4 + one parquet episode per rollout in
  lerobot v2.1 format
- Five-module diagnostic framework that compares a candidate eval dataset against the
  reference training dataset

What's **planned** (not wired yet):
- RLinf-driven PPO/GRPO post-training loop. `third_party/RLinf` is checked in but not
  invoked by anything in `src/` yet — superseded in practice by `src/rl/simple/`
  (single-process PPO, see results below).

---

## Eval results — pick_orange (SO-101)

`src/rl/simple/` is a single-process PPO on a frozen π₀.₅ + small Gaussian residual
head. 200 PPO iters (~1.0 M env steps, single 4090, ~7 h). All numbers below:
n=60 eps, num_envs=12, fast<900 means success within 30 s of sim time, failA=0 means
no collision / drop (all failures are timeouts).

**Two time budgets side-by-side.** Same ckpt, same `simple/eval.py`; the only diff is
`--episode-length-s` (and matching `--max-ep-steps`). 90 s is the budget the original
`src/eval.py` actually used (verified via `meta/episodes.jsonl` max-len 2700 @ 30 fps),
and is what the documented 60 % SFT baseline was scored under.

| ckpt | 45 s succ | 45 s fast<900 | 90 s succ | 90 s fast<900 |
|---|---|---|---|---|
| **SFT baseline** (zero residual head) | 28.33 % (17/60) | 6.67 % | **56.67 %** (34/60) | 28.33 % |
| **RL iter100** (90 s = avg of 2 runs) | 53.33 % (32/60) | 35.00 % | **72.50 %** (avg 73.33 / 71.67) | 43.33 % |
| **RL iter200** | 61.67 % (37/60) | 45.00 % | **68.33 %** (41/60) | 36.67 % |

- **Doc 60 % baseline strictly reproduced**: SFT v3 56.67 % vs documented 60.0 % —
  gap 3.3 pp, within 1 σ (~6 pp) for n=60.
- **RL lift survives the strict 90 s budget**: iter100 +15.83 pp, iter200 +11.66 pp
  over SFT v3.
- **iter100 > iter200 reversal at 90 s** (vs iter200 > iter100 at 45 s): ~4 pp gap is
  at the noise edge — either mild long-tail overfit on iter200, or seed noise. Needs
  ≥3 seeds to disambiguate.
- **Budget bonus 45 s → 90 s**: SFT +28.34 pp (almost entirely from the long tail —
  matches EXP_03 length-inflation 2.735× from `docs/sft_diagnostics_findings.md`);
  iter200 only +6.66 pp (short-ep success already saturated).

Full report: [docs/simple_ppo_step9_report.md](docs/simple_ppo_step9_report.md).
Training entry: `src/rl/simple/train.py`. Eval entry: `src/rl/simple/eval.py`.

---

## Repository Layout

```
lerobot-rlinf/
├── README.md / README.zh.md
├── ARCHITECTURE.zh.md / PROGRESS.zh.md          # WIP design docs (placeholders)
├── pyproject.toml                                # name=lerobot-rlinf, src layout
├── .gitmodules                                   # three submodules under third_party/
├── docs/
│   ├── notes.md / notes.zh.md                    # canonical setup/gotchas reference
│   ├── todo.md
│   ├── sft_diagnostics_findings.md               # applied findings from src/diagnostics
│   └── simple_ppo_step9_report.md                # simple PPO 200-iter train + eval report
├── assets/so_arm100/                             # SO-101 URDF + Isaac USD imports
├── src/
│   ├── sft_train.py     # → openpi train.py (LoRA π₀.₅)
│   ├── eval.py          # two-process orchestrator: openpi server + eval_run client
│   ├── eval_run.py      # headless Isaac Lab loop (video + lerobot v2.1 dataset out)
│   ├── tb_tailer.py     # mirrors openpi train.log → TensorBoard scalars
│   ├── diagnostics/     # ref-vs-candidate dataset diagnostic CLI (python -m src.diagnostics)
│   │   ├── __main__.py  base.py  registry.py  result.py  schema.py
│   │   ├── io.py        report.py
│   │   └── modules/     # episode_length, action_smoothness, compounding_error,
│   │                    # mode_averaging, state_coverage
│   └── rl/
│       ├── envs/                                # Isaac Lab env wrapper + OOD KD-tree
│       │   ├── isaaclab_pick_orange.py          # subprocess IPC + sparse 3-stage reward
│       │   └── ood_kdtree.py                    # demo-manifold KNN penalty (EXP_05 hook)
│       └── simple/                              # single-process PPO post-training (replaces rlinf shell)
│           ├── config.py        policy.py       # ResidualGaussianPolicy (frozen π₀.₅ + residual head)
│           ├── rollout_buffer.py ppo.py         # GAE + clipped surrogate + value clip
│           ├── reward_shaping.py                # OOD penalty + survival cost + dense lift
│           ├── bc_anchor.py                     # demo BC anchor (off by default in step9)
│           ├── _openpi_server.py                # shared serve_policy process lifecycle
│           ├── train.py                         # train entry (jsonl + TB + ckpt)
│           └── eval.py                          # offline eval entry (n_episodes × num_envs)
└── third_party/
    ├── openpi/          (EverNorif fork, branch lerobot-v0.3.3) — train + serve
    ├── leisaac/         (LightwheelAI)                          — Isaac Lab tasks
    └── RLinf/           (woshinideba1425 fork)                  — reserved for Phase 3
```

---

## The two-venv split (and why)

Eval has to talk to two stacks that can't share a venv. **openpi** is built on JAX and
pins `numpy >= 2`; **Isaac Sim 5.1** pins `numpy == 1.26`. So:

- `third_party/openpi/.venv/` — built via `uv sync` inside the openpi submodule.
  Owns: JAX, optax, π₀.₅ weights, `serve_policy.py`, `train.py`, `compute_norm_stats.py`.
- `rlinf-isaacsim-env` (conda) — has Isaac Sim 5.1 + Isaac Lab 2.x + the editable
  `leisaac` install. Owns: `eval_run.py` runtime.

`src/eval.py` is the only place these meet: it spawns `openpi/scripts/serve_policy.py`
in the openpi venv, waits for the WebSocket port, then spawns `eval_run.py` in the
Isaac Sim venv. Same applies to training — `sft_train.py` shells out to the openpi
venv directly, so you can run it from any shell.

See [docs/notes.md](docs/notes.md) for the full set of install fiddles.

---

## Robot: SO-ARM100 / SO-101

- 6 DOF (5 joints + 1 gripper)
- Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- Workspace radius ~20 cm (vs Franka's ~80 cm — tighten any command/spawn ranges copied
  from Franka tasks)
- Zero pose **is** the standing pose (non-zero `init_state.joint_pos` will flatten the arm)
- EE body for command/reward: `gripper_frame_link`

Joint limits and PD tuning gotchas: [docs/notes.md](docs/notes.md).

---

## Usage

### One-time setup

```bash
git submodule update --init --recursive

# openpi venv (uv-based)
cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && cd ../..

# leisaac editable install in the Isaac Sim conda env
conda activate rlinf-isaacsim-env
pip install -e ./third_party/leisaac/source/leisaac --no-deps
pip install -e .
```

### SFT — LoRA π₀.₅ on pick-orange (60 demos)

```bash
# HF_TOKEN if the dataset/weights need auth
source .env

python src/sft_train.py --exp-name=so101_pick_orange_lora_v0
```

The default config `pi05_lora_so101_pick_orange` lives in EverNorif's openpi fork at
`third_party/openpi/src/openpi/training/config.py`. Checkpoints land in
`outputs/<config>/<exp>/<step>/`. First invocation downloads `pi05_base` (~5–10 GB) from
`gs://openpi-assets` into `~/.cache/openpi`.

### TensorBoard sidecar

openpi only logs to wandb. To mirror scalars into TensorBoard:

```bash
python src/tb_tailer.py /path/to/train.log /path/to/tb_logdir
tensorboard --logdir /path/to/tb_logdir --port 6006
```

It parses `Step N: grad_norm=… loss=… param_norm=…` lines and recomputes the LR from
openpi's warmup-cosine schedule with the same defaults.

### Evaluation in Isaac Lab

```bash
python src/eval.py --exp-name=so101_pick_orange_lora_v0 --eval-rounds=20
```

This spawns `openpi serve_policy.py` (loads the checkpoint, JIT-compiles the model,
binds port 8000), waits for the port, then runs `eval_run.py` headless inside the
Isaac Sim venv. Each episode writes:

- `videos/chunk-000/observation.images.{front,wrist}/episode_NNNNNN.mp4`
- `data/chunk-000/episode_NNNNNN.parquet` with columns matching EverNorif's reference dataset

`--prefetch` is **off by default** — async chunk prefetch hides infer latency but the
policy plans from a 7-step-stale obs and visibly jerks at chunk boundaries. Re-enable
only after implementing receding-horizon execution / temporal ensembling.

### Dataset diagnostics

Compare an eval-rollout dataset against the training reference:

```bash
python -m src.diagnostics \
  --ref  /home/hlei/.cache/huggingface/lerobot/EverNorif/leisaac-pick-orange \
  --cand outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999/dataset \
  --out-json /tmp/diag.json \
  --out-md   /tmp/diag.md
```

Five plug-in modules, each one a `Diagnostic` subclass registered via
`@register_diagnostic(...)`:

| Module                       | What it answers                                         |
|------------------------------|---------------------------------------------------------|
| EXP_01 Mode Averaging        | Are action distributions collapsing (L2 BC mode covering)? |
| EXP_02 Compounding Error     | Are joint-space arc lengths inflating per frame?        |
| EXP_03 Episode-Length Inflation | Are candidate episodes longer than reference?        |
| EXP_04 Action Smoothness     | Are there chunking / EMA artifacts in the action stream? |
| EXP_05 State Coverage Divergence | Do candidate joint configs leave the demo manifold? |

See [docs/sft_diagnostics_findings.md](docs/sft_diagnostics_findings.md) for a worked
example: ckpt 24999 with 60–70% success rate, EXP_03 + EXP_05 jointly CRITICAL → policy
stalls in OOD joint regions (not action squashing, not compounding error).

`python -m src.diagnostics --list` enumerates registered modules; `--selftest` runs an
end-to-end smoke test.

---

## Stack & Versions

| Component   | Version                          | Notes |
|-------------|----------------------------------|-------|
| Isaac Sim   | 5.1.0                            | pip install, Python 3.11 |
| Isaac Lab   | 2.x (paired w/ Isaac Sim 5.1)    | bundled inside leisaac submodule |
| openpi      | EverNorif fork, `lerobot-v0.3.3` | LoRA π₀.₅ kernel; train.py + serve_policy.py |
| leisaac     | LightwheelAI main                | LeIsaac-SO101-PickOrange-v0 task |
| LeRobot     | v2.1 dataset format              | embedded in openpi's data pipeline |
| RLinf       | woshinideba1425 fork             | reserved for Phase 3 — not invoked yet |
| PEFT        | bundled by openpi                | rank-16 LoRA on action-expert `q/k/v/o_proj` |
| PyTorch     | bundled with Isaac Sim 5.1       | tb_tailer.py only |

---

## obs/action contract (current SFT path)

Matches `EverNorif/leisaac-pick-orange` exactly — must, since the model was trained
on it:

```python
# parquet columns written by eval_run.py
action            fixed_size_list<float32>[6]   # motor degrees, joint order above
observation.state fixed_size_list<float32>[6]   # motor degrees
timestamp         float32                       # seconds
frame_index, episode_index, index, task_index   int64
```

Image streams (mp4): `observation.images.front`, `observation.images.wrist`,
encoded at 30 fps with libx264 (codec deviation vs reference av1; same container,
lerobot's pyav reader is codec-agnostic).

---

## Project status

| Phase | What | Status |
|-------|------|--------|
| 1 | SO-101 URDF → Isaac Sim USD, env smoke tests             | done |
| 2 | LoRA π₀.₅ SFT on pick-orange + leisaac eval + diagnostics | live (this repo) |
| 3 | RLinf-driven PPO/GRPO post-training                      | planned — RLinf submodule is checked in but `src/` does not invoke it |

Open items: [docs/todo.md](docs/todo.md). Setup gotchas catalogue: [docs/notes.md](docs/notes.md).
