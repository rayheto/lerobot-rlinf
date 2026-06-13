# lerobot-rlinf

**SO-101 + π₀.₅ 后训练胶水层，当前处于 SFT 阶段。**
将 `openpi` 的 LoRA 训练与 `leisaac` 的 Isaac Lab 评测作为两个协作进程驱动，
另带一套仅依赖数据集的诊断框架，用于闭环训练质量分析。

> English version: [README.md](README.md)

---

## 这个仓库实际上是什么

三套小型 Python 子系统，搭在三个 vendored 子模块之上——重活由子模块完成（训练核、
仿真器、机器人模型），本仓库为 SO-ARM100 / SO-101 机械臂把它们粘起来，并在最上层加一层分析。

```
                       ┌─────────────────────────────────────────┐
   sft_train.py  ───►  │  third_party/openpi  (LoRA π₀.₅ 训练核) │
                       └─────────────────────────────────────────┘
                                       │ SFT checkpoints/
                                       ▼
   eval.py ──► 拉起 ──► openpi serve_policy.py  (WebSocket 服务)
            └► 拉起 ──► eval_run.py ──► third_party/leisaac (Isaac Lab 环境)
                                                       │
                                                       ▼
                            lerobot v2.1 格式数据集 (data/ + videos/ + meta/)
                                                       │
                                                       ▼
                          python -m src.diagnostics  (参考集 vs 候选集)
                                                       │
                                                       ▼
                                  诊断报告 (.json + .md)
                                                       │
                                       诊断结论驱动 reward 设计
                                                       ▼
   src/rl/simple/train.py  ──► 冻结 π₀.₅ (serve_policy) + 高斯残差头
                                                       │  PPO + 针对 EXP_03/05 的
                                                       │  reward shaping（OOD 惩罚、
                                                       │  survival cost、dense lift）
                                                       ▼
                              残差头 ckpt  ──► src/rl/simple/eval.py
```

**当前已落地**：
- 在 LightwheelAI/leisaac-pick-orange 数据集上做 LoRA π₀.₅ 的 SFT（监督微调）
- 无头 Isaac Lab 评测，每条 rollout 输出一支 mp4 + 一份 lerobot v2.1 格式的 parquet
- 五模块诊断框架，比较候选评测数据集与训练参考数据集

**已规划但尚未接入**：
- RLinf 驱动的 PPO/GRPO 后训练主循环。`third_party/RLinf` 已经作为子模块入库，
  但 `src/` 下没有任何代码调用它 —— 实际由 `src/rl/simple/`（单进程 PPO，见下方
  评测成果）替代承担。

---

## 评测成果 —— pick_orange (SO-101)

一段成功的 RL rollout（3 个橘子依次抓起 → 放进盘子 → 末端归位）：

https://github.com/rayheto/lerobot-rlinf/raw/develop/docs/attachments/success_example.mp4

<sub>源文件：[docs/attachments/success_example.mp4](docs/attachments/success_example.mp4)</sub>

`src/rl/simple/` 是一套单进程 PPO，冻结 π₀.₅ + 一个小的高斯残差头。200 PPO iter
（约 1.0 M env steps，单 4090，约 7 小时）。下表所有数字：n=60 集，num_envs=12，
fast<900 表示 30 s 仿真时间内成功，failA=0 即没有任何碰撞/掉落，失败全是超时。

**两种时间预算并列展示**。同一份 ckpt、同一份 `simple/eval.py`，唯一变量是
`--episode-length-s`（及对应的 `--max-ep-steps`）。90 s 是原 `src/eval.py` 实际
跑的预算（由 `meta/episodes.jsonl` 中最长 2700 帧 @ 30 fps 验证），也是文档里
60 % SFT baseline 当时所用的预算。

| ckpt | 45 s succ | 45 s fast<900 | 90 s succ | 90 s fast<900 |
|---|---|---|---|---|
| **SFT baseline**（零残差头） | 28.33 % (17/60) | 6.67 % | **56.67 %** (34/60) | 28.33 % |
| **RL iter100**（90 s 取 2 次均值） | 53.33 % (32/60) | 35.00 % | **72.50 %**（73.33 / 71.67 均值） | 43.33 % |
| **RL iter200** | 61.67 % (37/60) | 45.00 % | **68.33 %** (41/60) | 36.67 % |

- **文档 60 % baseline 严格复刻**：SFT v3 56.67 % vs 文档 60.0 %，偏差 3.3 pp，在
  n=60 的 1 σ ≈ 6 pp 内。
- **RL 净提升在严格 90 s 预算下依旧成立**：iter100 vs SFT v3 **+15.83 pp**，
  iter200 **+11.66 pp**。
- **iter100 > iter200 在 90 s 下翻转**（45 s 下是 iter200 > iter100）：~4 pp 差落在
  噪声边缘 —— 可能 iter200 略 overfit 长尾，也可能 60 样本 seed 噪声，要 ≥3 seed
  才能判定。
- **45 s → 90 s 预算红利**：SFT +28.34 pp（几乎全来自长尾，呼应
  `docs/sft_diagnostics_findings.md` 里 EXP_03 时长劣化 2.735×）；iter200 仅
  +6.66 pp（短 ep 成功率已基本拿满）。

完整报告：[docs/simple_ppo_step9_report.md](docs/simple_ppo_step9_report.md)。
训练入口：`src/rl/simple/train.py`，评测入口：`src/rl/simple/eval.py`。

---

## 仓库结构

```
lerobot-rlinf/
├── README.md / README.zh.md
├── ARCHITECTURE.zh.md / PROGRESS.zh.md          # 设计文档草稿（占位）
├── pyproject.toml                                # name=lerobot-rlinf, src 布局
├── .gitmodules                                   # 三个 third_party/ 子模块
├── docs/
│   ├── notes.md / notes.zh.md                    # 配置 / 坑点的权威记录
│   ├── todo.md
│   ├── sft_diagnostics_findings.md               # 实际 SFT 落盘的诊断结论
│   └── simple_ppo_step9_report.md                # simple PPO 200-iter 训练 + 评测报告
├── assets/so_arm100/                             # SO-101 的 URDF + Isaac USD
├── src/
│   ├── sft_train.py     # → openpi train.py (LoRA π₀.₅)
│   ├── eval.py          # 两进程编排：openpi 推理服务 + eval_run 客户端
│   ├── eval_run.py      # 无头 Isaac Lab 推理回路（视频 + lerobot v2.1 数据集落盘）
│   ├── tb_tailer.py     # 把 openpi train.log 镜像为 TensorBoard scalars
│   ├── diagnostics/     # 参考集 vs 候选集 数据集诊断 CLI（python -m src.diagnostics）
│   │   ├── __main__.py  base.py  registry.py  result.py  schema.py
│   │   ├── io.py        report.py
│   │   └── modules/     # episode_length, action_smoothness, compounding_error,
│   │                    # mode_averaging, state_coverage
│   └── rl/
│       ├── envs/                                # Isaac Lab env wrapper + OOD KD-tree
│       │   ├── isaaclab_pick_orange.py          # subprocess IPC + sparse 3-stage reward
│       │   └── ood_kdtree.py                    # demo-manifold KNN penalty（接 EXP_05）
│       └── simple/                              # 单进程 PPO 后训练（取代 rlinf 外壳）
│           ├── config.py        policy.py       # ResidualGaussianPolicy（冻结 π₀.₅ + 残差头）
│           ├── rollout_buffer.py ppo.py         # GAE + clipped surrogate + value clip
│           ├── reward_shaping.py                # OOD penalty + survival cost + dense lift
│           ├── bc_anchor.py                     # demo BC anchor（当前 step9 默认关）
│           ├── _openpi_server.py                # 共享 serve_policy 进程生命周期
│           ├── train.py                         # 训练入口（jsonl + TB + ckpt）
│           └── eval.py                          # 离线评测入口（n_episodes × num_envs）
└── third_party/
    ├── openpi/          (EverNorif fork, branch lerobot-v0.3.3) — train + serve
    ├── leisaac/         (LightwheelAI)                          — Isaac Lab tasks
    └── RLinf/           (woshinideba1425 fork)                  — Phase 3 预留
```

---

## 双 venv 切分（以及为什么必须切）

评测要同时跟两个不能共用 venv 的栈打交道：**openpi** 基于 JAX，固定 `numpy >= 2`；
**Isaac Sim 5.1** 固定 `numpy == 1.26`。所以：

- `third_party/openpi/.venv/` — 在 openpi 子模块内用 `uv sync` 构建。
  归它管：JAX、optax、π₀.₅ 权重、`serve_policy.py`、`train.py`、`compute_norm_stats.py`。
- `rlinf-isaacsim-env`（conda）— 装着 Isaac Sim 5.1 + Isaac Lab 2.x + `leisaac` 的
  editable 安装。归它管：`eval_run.py` 的运行时。

`src/eval.py` 是这两侧唯一的交点：它在 openpi venv 下拉起 `openpi/scripts/serve_policy.py`，
等 WebSocket 端口起来后，再在 Isaac Sim venv 下拉起 `eval_run.py`。训练侧同理——
`sft_train.py` 直接调用 openpi venv 的 python，所以从任何 shell 都能跑，不需要先 activate。

完整的安装坑表见 [docs/notes.md](docs/notes.md)。

---

## 机器人：SO-ARM100 / SO-101

- 6 自由度（5 关节 + 1 夹爪）
- 关节顺序：`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- 工作空间半径约 20 cm（vs Franka 的约 80 cm —— 从 Franka 任务移植过来的指令/生成范围都要收紧）
- **零位即站立位**（设置非零的 `init_state.joint_pos` 反而会让手臂塌下来）
- 末端执行器 body（command/reward 用）：`gripper_frame_link`

关节限位与 PD 调参的坑：[docs/notes.md](docs/notes.md)。

---

## 使用

### 一次性安装

```bash
git submodule update --init --recursive

# openpi venv（uv 管理）
cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && cd ../..

# 在 Isaac Sim conda env 内做 leisaac 的 editable 安装
conda activate rlinf-isaacsim-env
pip install -e ./third_party/leisaac/source/leisaac --no-deps
pip install -e .
```

### SFT —— pick-orange 上的 LoRA π₀.₅（60 条示教）

```bash
# 数据集 / 权重需鉴权时记得 source HF_TOKEN
source .env

python src/sft_train.py --exp-name=so101_pick_orange_lora_v0
```

默认配置 `pi05_lora_so101_pick_orange` 定义在 EverNorif 的 openpi fork 里
（`third_party/openpi/src/openpi/training/config.py`）。checkpoint 落到
`outputs/<config>/<exp>/<step>/`。首次运行会从 `gs://openpi-assets` 下载
`pi05_base`（约 5–10 GB）到 `~/.cache/openpi`。

### TensorBoard 边车

openpi 只往 wandb 写日志。要把 scalar 同时镜像到 TensorBoard：

```bash
python src/tb_tailer.py /path/to/train.log /path/to/tb_logdir
tensorboard --logdir /path/to/tb_logdir --port 6006
```

它会解析 `Step N: grad_norm=… loss=… param_norm=…` 这类行，并按 openpi
默认的 warmup-cosine schedule 反算 lr。

### Isaac Lab 里的评测

```bash
python src/eval.py --exp-name=so101_pick_orange_lora_v0 --eval-rounds=20
```

会：拉起 `openpi serve_policy.py`（加载 checkpoint、JIT 编译、绑 8000 端口）→
等端口就绪 → 在 Isaac Sim venv 下无头跑 `eval_run.py`。每条 episode 落盘：

- `videos/chunk-000/observation.images.{front,wrist}/episode_NNNNNN.mp4`
- `data/chunk-000/episode_NNNNNN.parquet`，列与 EverNorif 参考数据集一一对齐

`--prefetch` **默认关闭** —— 异步预取下一段 action chunk 能隐藏推理延迟，
但策略会基于约 7 步前的观测做规划，每个 chunk 边界都会肉眼可见地抖一下。
等实现了 receding-horizon 执行 / temporal ensembling 之后再打开。

### 数据集诊断

把评测 rollout 的数据集对训练参考集做对照：

```bash
python -m src.diagnostics \
  --ref  /home/hlei/.cache/huggingface/lerobot/EverNorif/leisaac-pick-orange \
  --cand outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999/dataset \
  --out-json /tmp/diag.json \
  --out-md   /tmp/diag.md
```

五个插件模块，每个都是 `Diagnostic` 的子类，通过 `@register_diagnostic(...)` 注册：

| 模块                          | 回答的问题                                    |
|-------------------------------|-----------------------------------------------|
| EXP_01 Mode Averaging         | 动作分布是否被 L2 BC 压扁（mode covering）？  |
| EXP_02 Compounding Error      | 关节空间的每帧弧长是否被放大？                |
| EXP_03 Episode-Length Inflation | 候选 episode 是否比参考集明显变长？         |
| EXP_04 Action Smoothness      | action 流是否有 chunking / EMA 伪迹？         |
| EXP_05 State Coverage Divergence | 候选关节构型是否跑出了示教流形？           |

实际案例：[docs/sft_diagnostics_findings.md](docs/sft_diagnostics_findings.md) ——
ckpt 24999 成功率 60–70%，EXP_03 与 EXP_05 联合 CRITICAL → 策略陷在 OOD 关节区域
（不是动作被压扁，也不是 compounding error 把路径拉长）。

`python -m src.diagnostics --list` 列出已注册模块；`--selftest` 跑端到端冒烟测试。

---

## 软件栈

| 组件        | 版本                              | 说明 |
|-------------|-----------------------------------|------|
| Isaac Sim   | 5.1.0                             | pip 安装，Python 3.11 |
| Isaac Lab   | 2.x（与 Isaac Sim 5.1 配套）      | 打包在 leisaac 子模块里 |
| openpi      | EverNorif fork, `lerobot-v0.3.3`  | LoRA π₀.₅ 训练核；train.py + serve_policy.py |
| leisaac     | LightwheelAI main                 | LeIsaac-SO101-PickOrange-v0 任务 |
| LeRobot     | v2.1 数据集格式                   | 内嵌在 openpi 的数据流水线中 |
| RLinf       | woshinideba1425 fork              | 为 Phase 3 预留 —— 当前未被调用 |
| PEFT        | openpi 自带                        | action expert 上 rank-16 LoRA，目标 `q/k/v/o_proj` |
| PyTorch     | 跟随 Isaac Sim 5.1                | 仅 tb_tailer.py 用 |

---

## obs / action 契约（当前 SFT 路径）

与 `EverNorif/leisaac-pick-orange` 严格一致 —— 必须一致，模型是在它上面训的：

```python
# eval_run.py 写出的 parquet 列
action            fixed_size_list<float32>[6]   # 电机度数，按上方关节顺序
observation.state fixed_size_list<float32>[6]   # 电机度数
timestamp         float32                       # 秒
frame_index, episode_index, index, task_index   int64
```

图像流（mp4）：`observation.images.front`、`observation.images.wrist`，
30 fps libx264 编码（codec 与参考集的 av1 不同；容器一致，lerobot 的 pyav reader
对 codec 不挑食）。

---

## 项目进度

| 阶段 | 内容                                                       | 状态 |
|------|------------------------------------------------------------|------|
| 1 | SO-101 URDF → Isaac Sim USD、环境冒烟测试                   | 完成 |
| 2 | pick-orange 上的 LoRA π₀.₅ SFT + leisaac 评测 + 诊断框架    | 当前所在（本仓库） |
| 3 | RLinf 驱动的 PPO/GRPO 后训练                                | 规划中 —— RLinf 子模块已入库，但 `src/` 还没调用 |

待办：[docs/todo.md](docs/todo.md)。配置坑表：[docs/notes.md](docs/notes.md)。
