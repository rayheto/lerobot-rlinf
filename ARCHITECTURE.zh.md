# 项目结构与环境

本文件解释本仓库的代码组织、依赖三个外部仓库的方式，以及三套 Python
环境的分工。具体阶段进度看 [PROGRESS.zh.md](PROGRESS.zh.md)，调试踩坑
看 [docs/notes.zh.md](docs/notes.zh.md)，TODO 看 [docs/todo.md](docs/todo.md)。

最后更新：2026-06-02

---

## 一、项目角色与边界

本仓库（`lerobot-rlinf`）只做一件事：**给 SO-101 机械臂的 Pi 0.5 RL
后训练做 glue 层** —— 把第三方的 Isaac Lab env（LightwheelAI/leisaac）、
LeRobot SFT 产物、RLinf 训练框架串起来。它**不实现 env、也不实现训练
循环**：env 来自 leisaac，训练交给 RLinf。

> **历史**：2026-06-02 之前，本仓库还自己实现了 SO-101 sponge-bowl env
> （手搓 USD + CameraCfg + reward shaping）。Phase 2 eval 暴露 cross-domain
> gap（视觉/动力学/sim 域 0 数据三层 mismatch）后，env 层全部迁移到
> [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)（自带
> 配套 LeRobot 数据集和视觉对齐的 sim env）。详见 PROGRESS.zh.md 六、变更历史。

整条管线的代码物理分布在四个仓库（leisaac 是 2026-06-02 之后新增的）：

```
┌─────────────────────────────────────────────────────────────────────┐
│  lerobot-rlinf（本仓库，glue only）                                   │
│  ─────────────────────────                                           │
│  - ckpt remap 工具（lerobot → openpi）                                │
│  - SFT 启动脚本（调 lerobot-train CLI）                                │
│  - standalone eval driver（绕过 Ray 验证链路）                          │
│  - env / dataset / 训练全部不在本仓库实现                              │
└────┬───────────────────┬───────────────────────┬────────────────────┘
     │ import leisaac    │ bash scripts/…        │ python scripts/…
     │ （触发 gym 注册） │                       │
     ▼                   ▼                       ▼
┌──────────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
│ leisaac              │ │ lerobot 0.4.4   │ │ RLinf                │
│ (third_party/        │ │ (pypi)          │ │ (/home/hlei/RLinf)   │
│  leisaac, submodule) │ │ ─────────────   │ │ ─────────────        │
│ ─────────────────    │ │ - LeRobot       │ │ - SFT/PPO worker     │
│ - SO-101 ArtCfg      │ │   dataset/CLI   │ │ - openpi 模型加载      │
│ - lift_cube /        │ │ - Pi 0.5 模型    │ │ - FSDP 包装           │
│   pick_orange /      │ └─────────────────┘ │ - env wrapper        │
│   cleanup_trash 等   │                     │   (so101_lift.py)    │
│ - LeRobot recorder   │                     └──────────┬───────────┘
│ - 配套真机数据集（HF）│                                │
└──────────┬───────────┘                                │
           │ import isaaclab.envs:ManagerBasedRLEnv     │
           ▼                                            │
┌─────────────────────────────────────────────────────────────────────┐
│  IsaacLab 2.3.0（leisaac 嵌套的 submodule）                           │
│  third_party/leisaac/dependencies/IsaacLab/                          │
│  ─────────────────────────────────────                              │
│  - ManagerBasedRLEnv 框架                                            │
│  - Isaac Sim 5.1 + PhysX 仿真后端                                    │
└─────────────────────────────────────────────────────────────────────┘
```

设计原则：
- **本仓库不写 env、不写训练代码**。env 来自 leisaac（Apache-2.0 vendored
  submodule）；SFT 调 lerobot CLI；RL 训练在 RLinf。
- **gym env 注册通过 `import leisaac` 副作用触发**。RLinf / eval driver
  只需在启动前 `import leisaac`，就能 `gym.make("LeIsaac-SO101-LiftCube-v0")`。
- **leisaac 资产路径用 `LEISAAC_ASSETS_ROOT` 环境变量定位**到
  `third_party/leisaac/assets/`（缺省 `git rev-parse` 会指错位置）。
- **跨仓库写依赖通过 conda env 的 `pip install -e .` 落地**，不靠
  PYTHONPATH 黑魔法。

---

## 二、目录结构

```
lerobot-rlinf/
├── pyproject.toml              # name=lerobot-rlinf, version=0.0.2（glue only）
├── PROGRESS.zh.md              # 阶段进度 + 风险登记
├── ARCHITECTURE.zh.md          # 本文件
├── README.md / README.zh.md    # 安装 & quickstart
├── .gitmodules                 # third_party/leisaac
│
├── src/lerobot_rlinf/          # 缩水成空命名空间（env 层迁去 leisaac）
│   ├── __init__.py
│   ├── assets/__init__.py      # 占位说明
│   └── tasks/__init__.py       # 占位说明
│
├── scripts/
│   ├── smoke_lift_cube_leisaac.py   # leisaac env smoke
│   ├── eval_pi05_liftcube.py        # standalone eval（绕过 Ray，跑新 env）
│   ├── audit_ckpt_keys.py           # Phase 1.5：lerobot ↔ openpi key 对齐
│   ├── convert_lerobot_to_openpi.py # Phase 1.5：ckpt remap
│   ├── extract_norm_stats.py        # Phase 1.5：norm_stats 真值替换 dummy
│   ├── convert_urdf_to_usd.py       # 通用工具
│   ├── inspect_so101.py             # USD/joint 验证（保留）
│   ├── sft_pi05_sponge.sh           # lerobot-train CLI（sponge baseline，历史）
│   ├── sft_smoke.sh                 # 2-step smoke
│   ├── sft_run_with_tb.sh           # sft + tensorboard
│   └── _sft_tb_tail.py              # tb 监控
│
├── third_party/
│   └── leisaac/                # git submodule, recursive
│       ├── source/leisaac/     # pip install -e（含 LeIsaac-SO101-* gym IDs）
│       ├── assets/             # USD 资产（人工下载）
│       │   ├── robots/so101_follower.usd     # 23 MB，HF leisaac_env
│       │   └── scenes/table_with_cube/       # 5 MB，GitHub release v0.1.2
│       │       ├── scene.usd
│       │       ├── cube/  textures/
│       └── dependencies/IsaacLab/  # 嵌套 submodule，IsaacLab 2.3.0
│
├── outputs/                          # 训练产物（.gitignore）
│   ├── sft_pi05_sponge/
│   │   └── checkpoints/
│   │       ├── 004000/
│   │       ├── 006000/
│   │       ├── 008000/
│   │       └── 010000/               # ← 当前 lerobot SFT 最新 ckpt
│   │           ├── pretrained_model/
│   │           │   ├── model.safetensors  # 7.0G
│   │           │   ├── policy_preprocessor.json
│   │           │   ├── policy_preprocessor_step_2_normalizer_processor.safetensors
│   │           │   ├── policy_postprocessor.json
│   │           │   ├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
│   │           │   ├── config.json
│   │           │   └── train_config.json
│   │           └── training_state/
│   └── sft_pi05_sponge_tb/           # tensorboard logdir
│
└── docs/
    ├── notes.md / notes.zh.md        # 踩坑笔记（中英）
    └── todo.md                       # 子任务级 TODO
```

仓外但属于本管线的关键文件（**不在本仓库**，但本仓库的脚本/env 引用它们）：

```
/home/hlei/RLinf/                                # uv 管理
├── rlinf/
│   ├── envs/isaaclab/
│   │   ├── isaaclab_env.py                      # IsaaclabBaseEnv 基类
│   │   └── tasks/
│   │       ├── stack_cube.py                    # Franka 参考实现
│   │       └── so101_lift.py                    # 我们写的 SO-101 task wrapper
│   ├── workers/sft/fsdp_vla_sft_worker.py       # RLinf SFT worker（撞 R1）
│   └── models/embodiment/openpi/
│       └── dataconfig/
│           ├── isaaclab_so101_dataconfig.py    # 我们写的 dataconfig
│           │                                    # + 两个 lerobot/openpi
│           │                                    # 兼容 monkey-patch
│           └── __init__.py                      # 注册 pi05_isaaclab_so101_lift
│
├── examples/
│   ├── sft/
│   │   ├── train_vla_sft.py                     # RLinf SFT 入口
│   │   ├── run_vla_sft.sh                       # bash 包装
│   │   └── config/
│   │       └── so101_sponge_sft_openpi_pi05.yaml  # 我们写的 SFT yaml
│   └── embodiment/config/
│       ├── env/isaaclab_so101_sponge.yaml       # 我们写的 env yaml
│       └── isaaclab_so101_sponge_ppo_openpi_pi05.yaml  # 我们写的 PPO yaml
│                                                # （只 eval 和 PPO 都用它）
└── .venv/                                       # uv 创建（Python 3.11.14）
```

---

## 三、Python 环境（三套）

每个环境职责单一，互不串包。命令行触发哪个环境，看启动脚本第一行
shebang / `LEROBOT_BIN` 之类的硬编码路径。

### 1. `rlinf-isaacsim-env`（conda，Isaac Sim 5.1 + Isaac Lab）

- **路径**：`/home/hlei/miniconda3/envs/rlinf-isaacsim-env`
- **Python**：3.11.15
- **装了什么**：
  - `isaacsim==5.1.0.0` + 全套 `isaacsim-*` 扩展
  - `isaaclab` + `isaaclab_tasks` + ... 共 5 个 editable package，
    **指向 `third_party/leisaac/dependencies/IsaacLab/source/`**
    （2026-06-02 之前指向 `third_party/IsaacLab/`，已删；版本 2.3.0，
    从 0.54.3 downgrade，因为 leisaac 强 pin 此版本）
  - `leisaac`（editable，`third_party/leisaac/source/leisaac/`）——
    触发 `LeIsaac-SO101-*` 任务注册
  - `lerobot-rlinf`（本仓库 editable）—— 现仅提供命名空间
  - 仅做 env 侧依赖，**不装 lerobot / openpi**
- **什么时候用**：
  - `scripts/smoke_lift_cube_leisaac.py`、`scripts/eval_pi05_liftcube.py`
  - `scripts/inspect_so101.py`
  - RLinf 端调 Isaac Lab env 时由 RLinf 自动激活（component_placement
    机制；TODO：核对 RLinf 是否真的切到这个 env，还是装在自己的 .venv 里）
- **leisaac 资产位置**：脚本 import leisaac 前必须 `os.environ["LEISAAC_ASSETS_ROOT"] = ".../third_party/leisaac/assets"`，
  否则 leisaac 用 `git rev-parse --show-toplevel` 把 ASSETS_ROOT 解析到
  本仓库根目录的 `assets/`（已删），加载 USD 时报 `Failed to open layer`。
- **不要装的东西**：`lerobot`、`accelerate`、任何拖 numpy>=2 进来的包。
  `isaacsim-kernel==5.1.0.0` 锁 `numpy==1.26.0`，违反会导致几百个
  `isaacsim.*` 扩展加载失败（详见 `docs/notes.zh.md`）。

### 2. `rlinf-lerobot-train`（conda，lerobot SFT CLI）

- **路径**：`/home/hlei/miniconda3/envs/rlinf-lerobot-train`
- **Python**：3.11.15
- **装了什么**：
  - `lerobot==0.4.4`（pypi）
  - `transformers @ git+huggingface/transformers@fix/lerobot_openpi`
    （Pi 0.5 需要打过补丁的版本，`--no-deps` + `--force-reinstall`）
  - `huggingface_hub<0.36`（lerobot 0.4.4 pin）
  - 不装 isaacsim / openpi
- **什么时候用**：
  - `scripts/sft_pi05_sponge.sh`（主 SFT）
  - `scripts/sft_smoke.sh`（2-step smoke）
- **跑出来的产物**：`outputs/sft_pi05_sponge/checkpoints/<step>/pretrained_model/`
- **环境特性**：lerobot ckpt 用的就是这个环境的 PI05Policy 类序列化，
  反序列化时需要同一个 transformers fork，否则 import 阶段就崩。

### 3. RLinf 自己的 `.venv`（uv，RLinf 训练 + openpi 模型）

- **路径**：`/home/hlei/RLinf/.venv`（uv 在 RLinf 仓库根目录创建）
- **Python**：3.11.14
- **装了什么**（关键依赖）：
  - `rlinf==0.3.0`（editable）
  - `openpi==0.1.0` + `openpi.models_pytorch`（含我们之前 patch 过又
    撤回的 `gemma_pytorch.py`）
  - `lerobot==0.4.4`（被 openpi 间接拉，但有命名空间冲突 —— 我们写了
    `lerobot/common/__init__.py` shim 把老路径桥到新路径）
  - 我们的兼容 monkey-patch 在 `rlinf/models/embodiment/openpi/dataconfig/
    isaaclab_so101_dataconfig.py`：lerobot get_safe_version fallback +
    PromptFromLeRobotTask DataFrame→dict 转换
  - PyTorch 2.6（!! R1 的诱因之一）
- **什么时候用**：
  - RLinf SFT smoke（被搁置，详见 R1）
  - 后续 RLinf eval / PPO
- **怎么激活**：uv run 或 直接调 `.venv/bin/python`。RLinf 的脚本
  `examples/sft/run_vla_sft.sh` 里 `which python` 应该已经指过来。

### 三套 env 不互通的边界示意

```
rlinf-isaacsim-env   →  只跑 Isaac Lab / sim sanity
rlinf-lerobot-train  →  只跑 lerobot-train CLI
RLinf/.venv (uv)     →  跑 RLinf train_vla_sft.py / PPO
```

跨环境通信走文件系统：
- `outputs/sft_pi05_sponge/checkpoints/.../model.safetensors`
  （`rlinf-lerobot-train` 写 → 计划被 `RLinf/.venv` 读）
- `~/.cache/huggingface/`（共享数据集 / 预训练模型）
- `assets/so_arm100/.../*.usd`（`rlinf-isaacsim-env` 转换 → RLinf
  Isaac Lab worker 读）

---

## 四、关键 import 边界与注册副作用

理解这部分能避免 "环境装对了但 gym.make 报 UnregisteredEnv"
之类的迷惑。

### 4.1 gym env 注册路径

```python
# RLinf 端的 task wrapper（rlinf/envs/isaaclab/tasks/so101_lift.py）
# 在 _make_env_function() 里必须显式 import：
import lerobot_rlinf   # noqa: F401
# ↑ 这一行触发：
#   lerobot_rlinf/__init__.py
#   → src/lerobot_rlinf/tasks/__init__.py
#   → src/lerobot_rlinf/tasks/lift/__init__.py
#   → src/lerobot_rlinf/tasks/lift/config/so101/__init__.py
#   → 4 个 gym.register(...) 调用

import gymnasium as gym
env = gym.make("Isaac-Lift-Sponge-Bowl-SO101-v0", ...)
```

如果 `rlinf-isaacsim-env` 没装 `lerobot-rlinf`（editable），上面这个
import 会失败 → gym 找不到 ID → UnregisteredEnv。

### 4.2 openpi dataconfig 注册

```python
# RLinf 端 SFT/PPO 启动时会 import：
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
# 这个 import 副作用包括：
#   - 触发 isaaclab_so101_dataconfig.py 顶层的两个 monkey-patch
#     - _patch_lerobot_revision_fallback()
#     - _patch_openpi_prompt_from_lerobot_task()
#   - 注册 "pi05_isaaclab_so101_lift" config 名

# yaml 通过 actor.model.openpi.config_name 引用：
#   actor.model.openpi.config_name: "pi05_isaaclab_so101_lift"
```

monkey-patch 必须在主进程 import 阶段就执行，因为 PyTorch DataLoader
worker 用 spawn 启动，workers **不会重新执行** patch（详见 `docs/notes.zh.md`
"SFT 冒烟调试" 段）。

### 4.3 lerobot 0.1.0 → 0.4.4 路径迁移 shim

openpi 0.1.0 还在 import 老路径 `lerobot.common.datasets.*`，lerobot
0.4.4 已经移到 `lerobot.datasets.*`。我们在 `.venv/.../lerobot/common/__init__.py`
写了一个 shim：

```python
import sys as _sys
from lerobot import datasets as _new_datasets
_sys.modules.setdefault(f"{__name__}.datasets", _new_datasets)
```

仅在 `RLinf/.venv` 里生效，其他两个 env 不需要。

---

## 五、数据流（端到端）

按时间顺序：

1. **资产准备**
   - `assets/so_arm100/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`
   - `scripts/convert_urdf_to_usd.py` 转出 USD（在 `rlinf-isaacsim-env`）

2. **Env smoke**（W1，已完成）
   - `scripts/smoke_lift_so101.py` 在 `rlinf-isaacsim-env` 里跑
     `gym.make + reset + step`，验证 SO101_CFG 关节顺序、限位、PD 不
     爆炸

3. **lerobot SFT**（dry-run → 现在变成产 ckpt 的主路径）
   - `scripts/sft_pi05_sponge.sh` 在 `rlinf-lerobot-train` 里跑
     `lerobot-train` CLI，输入是 HF 的 `aswinkumar99/...-sponge-...`
     数据集，输出到 `outputs/sft_pi05_sponge/checkpoints/<step>/`

4. **Phase 0 sanity**（已完成）
   - `scripts/replay_dataset_actions.py` 在 `rlinf-isaacsim-env` 里跑，
     直接读数据集 parquet，把动作喂到 env，对比 env state 和数据集 state

5. **Phase 1.5 — lerobot ckpt 加载到 RLinf openpi**（待做）
   - key 审计 + normalizer 抽取 + remap 写入（在 RLinf/.venv 里）

6. **Phase 2 — RLinf only_eval**（待做）
   - `run_embodiment.sh isaaclab_so101_sponge_ppo_openpi_pi05`
     带 `only_eval: True`

7. **Phase 3 — RLinf PPO**（待做，先解 R1）

---

## 六、命名约定与不变量

- **关节顺序（6 DoF）**：
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
  本仓库 / RLinf wrapper / aswinkumar99 数据集都用这个顺序。**改顺序就全断**。
- **动作 / 状态单位**：度。**不是弧度**（即使 URDF 限位是弧度）。
  `JointPositionActionCfg` 内部会把度数转弧度喂 PhysX。
- **图像格式**：uint8 `[B, H, W, 3]`，分辨率 224×224。policy 内部归一化。
- **obs key 命名**：
  - env 出：`obs["policy"]` `[B,6]` + `obs["images"]["cam_high"]` +
    `obs["images"]["cam_wrist"]`
  - lerobot 数据集：`observation.images.overhead` + `observation.images.wrist`
    + `observation.state`
  - pi05_base policy：`observation.images.base_0_rgb` +
    `observation.images.right_wrist_0_rgb` + `observation.state`
  - 桥接靠 SFT CLI 的 `--rename_map`，RLinf 端 task wrapper 也要做同样
    的 rename
- **action_dim = 6**、**num_action_chunks = 50**。pi05_base 默认 chunk
  pad 到 32 维 action_dim、50 step；我们 finetune 后仍是 50 step、真实
  6 维。

---

## 七、如果要换任务（端到端 cost 估算）

复用：SO-101 资产、PD 整定、Isaac Lab 基类、RLinf SFT/PPO 脚本与
hparams、安装链路。

每任务的新增工作：
1. 新数据集（HF 拉公开的，或自己采）
2. 新 env config：`src/lerobot_rlinf/tasks/<task>/...`（fork lift/）
3. RLinf 端新 task wrapper：`rlinf/envs/isaaclab/tasks/<task>.py`（fork so101_lift.py）
4. 新 dataconfig：`isaaclab_<task>_dataconfig.py`
5. 3 个 yaml（env + sft + ppo）—— fork-and-edit

无新增基础设施成本，全部走当前模板。
