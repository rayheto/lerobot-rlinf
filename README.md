# lerobot-rlinf

**Post-training framework: RLinf + Isaac Sim + LeRobot Pi 0.5**
Simulation-based RL training stack for the SO-ARM100 / SO-101 arm.

> 中文版见 [README.zh.md](README.zh.md)

---

## Goal

Use RLinf as the training main loop, with the LeRobot Pi 0.5 VLA model (pretrained weights) as the policy, and Isaac Sim as the simulator. The policy is post-trained (LoRA-first) via PPO/GRPO against a target task — initial milestone: object grasping.

```
┌────────────────────────────────────────────────────────┐
│                  RLinf main loop                        │
│                                                         │
│  ┌──────────────┐  obs_dict   ┌──────────────────────┐ │
│  │ Isaac Sim Env│ ──────────► │  Pi 0.5 Actor        │ │
│  │ (SO-101)     │ ◄────────── │  (LeRobot + LoRA)    │ │
│  └──────────────┘  action      └──────────────────────┘ │
│         │                              │                │
│         └────── Trajectory Buffer ─────┘                │
│                       │                                  │
│                  PPO / GRPO                              │
└────────────────────────────────────────────────────────┘
```

**Core principle:** LeRobot contributes only the model architecture and pretrained weights. RLinf owns the training loop, data collection, and gradient updates end-to-end.

---

## Repository Layout (planned)

```
lerobot-rlinf/
├── README.md / README.zh.md
├── assets/                 # Robot assets
│   └── so_arm100/          # URDF / USD / meshes
├── envs/                   # Isaac Sim envs (OmniIsaacGymEnvs-compatible)
│   └── so101_pick_task.py
├── actors/                 # Pi 0.5 actor / critic wrappers
│   ├── pi05_actor.py
│   └── pi05_critic.py
├── trainers/               # RLinf loop + PPO adapter
├── configs/                # YAML / Hydra configs
├── scripts/                # Asset conversion, alignment tests, train entrypoints
└── third_party/            # Vendored / submodule deps
```

---

## Stack & Versions

| Component | Version | Notes |
|------|------|------|
| Isaac Sim | 4.2.0 | Seeed Wiki verified |
| OmniIsaacGymEnvs | Isaac Sim 4.2 branch | Vectorized env base |
| LeRobot | main | Source of Pi 0.5 |
| RLinf | main | Training loop framework |
| PEFT | ≥ 0.10.0 | LoRA |
| PyTorch | ≥ 2.3.0 | bf16 autocast |

---

## Robot: SO-ARM100 / SO-101

- 6 DOF (5 joints + 1 gripper)
- Source: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) → `Simulation/SO101/so101_new_calib.urdf`
- Imported into Isaac Sim via URDF Importer → USD, fixed base, joint-position drive

| Joint | Name | Range |
|------|------|------|
| 0 | shoulder_pan | ±π |
| 1 | shoulder_lift | ±π/2 |
| 2 | elbow_flex | ±π |
| 3 | wrist_flex | ±π/2 |
| 4 | wrist_roll | ±π |
| 5 | gripper | [0,1] normalized |

---

## obs Contract (must match LeRobot Pi 0.5 dataset)

```python
obs_dict = {
    "observation.images.cam_high":  Tensor[B, 3, 224, 224],  # float32, [0,1]
    "observation.images.cam_wrist": Tensor[B, 3, 224, 224],
    "observation.state":            Tensor[B, 12],  # 6 joint pos + 6 joint vel
}
```

Key naming must follow `observation.images.<cam_name>` exactly — LeRobot's internal embedding lookup keys off these strings.

---

## Post-training Strategy: LoRA

- **Freeze vision encoder** (SigLIP / PaliGemma backbone)
- **Insert LoRA into the action expert**: rank 16–64, target `q_proj/k_proj/v_proj/o_proj`
- **flow-matching log_prob** is the core PPO integration challenge — verify whether LeRobot exposes `compute_log_prob()`; otherwise approximate via `−flow_matching_loss`
- **External critic**: Pi 0.5 has no value head — use a lightweight independent MLP

---

## Milestones

| Week | Goal |
|------|------|
| W1 | Asset prep: URDF → Isaac Sim, `env.step()` without images |
| W2 | obs alignment: dual-camera GPU tensor path, Sim2Sim test passes |
| W3 | Rollout pipeline: Trajectory Buffer + critic + eval loop |
| W4 | PPO integration: log_prob solved, first full training run |
| W5–6 | Stabilization: dense reward, parallel envs, domain randomization |

Full plan: `/home/hlei/MemoryPalace/robotics/lerobot/rlinf_isaacsim_pi05.md`.

---

## Branching

- `main` — stable, reviewed PRs only
- `develop` — daily integration branch
- `feature/*` — feature branches

---

## Key Risks

| Risk | Mitigation |
|------|------|
| Pi 0.5 lacks `compute_log_prob()` | Start with REINFORCE; or flow-matching ELBO approximation |
| Multi-env camera OOM | Single env first; drop to 128×128 if needed |
| URDF physics off | Calibrate `drive_strength` / `damping` against real arm |
| action chunk vs single-step PPO | Take `chunk[0]`; revisit with multi-step PPO later |

---

## License

TBD. Upstream components retain their own licenses (LeRobot Apache-2.0; Isaac Sim proprietary SDK; SO-ARM100 — see upstream).
