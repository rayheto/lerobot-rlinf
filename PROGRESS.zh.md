# SO-101 Pi 0.5 RL 后训练 — 进度与计划

> 本文件追踪 W2–W4 路线图（SO-101 sponge-bowl 任务，sim-only）的实际进度，
> 和原 plan（`/home/hlei/.claude/plans/snoopy-percolating-gray.md`）的偏差，
> 以及当前已识别的风险。
>
> 调试细节请翻 `docs/notes.zh.md`，TODO 翻 `docs/todo.md`。

最后更新：2026-06-02

---

## 一、阶段进度（vs 原 plan）

| 阶段 | 原计划 | 实际状态 |
|---|---|---|
| **W1 — env** | SO-101 cube-lift + sponge-bowl Isaac Lab 环境 | ✅ **已完成**，commit 在 `develop` 分支（`fbcfa75` / `8d283ef`） |
| **Phase 0 — dataset action replay** | 跑 `scripts/replay_dataset_actions.py` 验证 dataset action 单位 == env action 单位 | ✅ **已跑（2026-06-01）**，结论见下（R2 已降级） |
| **Phase 1 — RLinf SFT**（原主路径） | 用 RLinf SFT 出 openpi-格式 ckpt | ❌ **撞墙**（见下文风险 R1） |
| **Phase 1' — lerobot SFT**（dry-run 路径） | 原计划只是验证数据健康度后丢弃 | ✅ **跑到 step 10000**，7.0G ckpt 落盘 `outputs/sft_pi05_sponge/checkpoints/010000/` |
| **决策转向**：跳过 RLinf SFT、走 lerobot ckpt → openpi remap | 不在原 plan | ✅ 2026-06-01 确认 |
| **Phase 1.5 — lerobot → openpi 加载链路** | 不在原 plan，决策转向后新增 | 🟡 **未开始**（下一步） |
| **Phase 1.5 — lerobot → openpi ckpt remap** | 不在原 plan | ✅ 已完成（保留为通用工具，未来 lift_cube SFT 复用） |
| **Phase 2 — `only_eval: True` 验证** | 用 SFT ckpt 跑 100 episodes，要求 ≥30% 成功率 | ❌ **不可达成**：cross-domain gap 三层 mismatch；pivot 见下 |
| **Pivot 2026-06-02** — 接入 LightwheelAI/leisaac 作为 env 层 | 不在原 plan | ✅ submodule 接入完成，smoke 通过；旧手搓 env 全删 |
| **Phase 2-bis — 在 leisaac lift_cube 数据集上重 SFT** | 替代 sponge 任务 | 🟡 未开始 |
| **Phase 3 — PPO/GRPO 后训练** | RL 后训练，目标 ≥80% | 🟡 未开始（待 R1 ViewBackward0 + Ray plumbing） |
| **Phase 5 — Sim2real** | 显式 deferred | 🟡 不在当前迭代 |

---

## 二、决策转向：为什么放弃 RLinf SFT 走 lerobot ckpt

原 plan 选 "RLinf 一统到底" 是为了避免跨 codebase 加载的 silent
`strict=False` 失败模式。但实际跑 RLinf SFT smoke 时撞到 PyTorch 2.6 +
openpi 0.1.0 + FSDP 的底层兼容 bug（详见 R1），所有 patch 尝试都失败。

**取舍**：
- 原方案：debug openpi 源码（不确定时长，可能踩新坑）
- 新方案：用 lerobot SFT 产物（已成功），写 lerobot → openpi weight remap

新方案的隐藏代价：**ViewBackward0 bug 没解决，只是延后到 Phase 3
PPO**。Phase 2 走 `only_eval`（inference-only，无 backward）可以绕过，
但 PPO 走同一份 `openpi/gemma_pytorch.py` backward 一定会再撞墙。

**取舍合理性**：先用 lerobot SFT 验证整条链路（数据 → policy →
sim eval）是通的，能拿 success rate，再花时间硬刚 openpi backward。
如果 lerobot ckpt 在 Phase 2 eval 表现就很差，那 Phase 3 也没必要开。

已撤回的临时 patch（保持环境干净）：
- `openpi/models_pytorch/gemma_pytorch.py` 中对 q/k/v 加 `.contiguous()` /
  `.clone()` / 禁用 gradient checkpointing force-enable —— 全撤回
- `examples/sft/config/so101_sponge_sft_openpi_pi05.yaml` 的
  `use_orig_params: True` —— 撤回为 False

---

## 三、当前已识别风险

按严重度排序。每条都说明**为什么是风险**、**触发条件**、**应对**。

### R1（🔴 高）：ViewBackward0 inplace 错误 — PPO 时必撞

**现象**：RLinf SFT smoke 跑到第一个 forward backward 就崩，traceback
落在 `openpi/models_pytorch/gemma_pytorch.py:176` 的 q_proj F.linear。
报错 `Output 0 of ViewBackward0 is a view and its base or another view
of its base has been modified inplace`。

**已排除的原因**：
- ✗ FSDP FlatParameter flatten（`use_orig_params=True` 后仍崩）
- ✗ FSDP mixed_precision dtype cast（`precision: null` → MixedPrecision
  全 None，没做 cast）
- ✗ q_proj 输入 view chain（`.clone()` + `.contiguous()` 都加了仍崩）
- ✗ openpi 强制开启的 gradient checkpointing（禁掉仍崩）

**剩余怀疑对象**：PyTorch 2.6 对 `torch.chunk` / multi-view tuple 返回的
inplace 检查变严，openpi `GemmaRMSNorm.forward` 里 `scale, shift, gate
= torch.chunk(modulation, 3, ...)` + 返回 tuple 命中。

**触发**：任何 backward through openpi gemma layers 的训练（SFT 或 PPO）。
**Inference-only**（`only_eval: True`）**不触发**。

**当前应对**：通过决策转向延后到 Phase 3。届时方案有：
1. 重写 `GemmaRMSNorm.forward` 为 out-of-place
2. 降级 PyTorch 到 2.4（已知该版本检查较宽松）
3. 降级或不用 FSDP（虽然已排除 FSDP 直接相关，但换个 wrapper 可能改变 view 链）

### R2（🟢 低，已降级）：dataset/env action 单位匹配 — 已验证

**为什么曾经是风险**：aswinkumar99 数据集的 action 是 Feetech 度数，
env 用 `JointPositionActionCfg`。如果单位不对（比如 deg 当 rad），
SFT 学到的 policy 在 env 里出 action 后机械臂动作完全错位。

**2026-06-01 重跑 `scripts/replay_dataset_actions.py`（ep0, 622 帧）结论**：
- 单位 ✅ 对齐（如果 deg/rad 错位会差 50×，实际差 5.69°，量级正确）
- 整体 stable mean err = 5.69°（脚本阈值 5° 触发 FAIL，仅差 0.69°）
- 关节级 mean err：shoulder_pan 1.5° / shoulder_lift 3.4° / elbow_flex 5.7°
  / **wrist_flex 11.4°** / wrist_roll 4.4° / gripper 7.7°
- 误差时间分布：t=250 / t=500 出现 mean 尖峰（21° / 32°），不是持续漂移

**根因**：**不是单位错位**，而是物理工作空间错配——env 里桌面高度
截断了数据集采集时允许的部分轨迹，机械臂触底卡 1-2 step。wrist_flex
的 11.4° 是 PD/重力压塌的稳态偏置（已知问题，详见 `docs/todo.md`
"env state ↔ 数据集 state 对齐" 段）。

**对后续阶段的影响**：
- 对 SFT 不致命：SFT 学的是数据集图像 + 动作，不依赖 env 跟踪精度
- 对 Phase 2 eval 不致命：用 success rate 当指标，不看 env state 误差
- 仅在 sim2real 阶段需要彻底对齐 workspace

**当前不修**。Phase 3 后如果 reward shaping 依赖姿态绝对值再回来处理。

### R3（🟠 中）：normalization 双层冲突

**为什么是风险**：lerobot ckpt 自带 `policy_preprocessor.json` +
`policy_preprocessor_step_2_normalizer_processor.safetensors`（state /
action 各有 mean/std）；RLinf openpi 期望读 `norm_stats.json`（之前为
SFT smoke 写过 dummy mean=0/std=1，**这是错的 stats**）。

两套 normalization 不能并存——必须把 lerobot 的 normalizer 提取出来转
成 openpi `norm_stats.json` 格式，否则 policy 输入分布全错，eval
success rate 会接近 0%。

**应对**：Phase 1.5 写 normalizer 抽取脚本，替换 dummy norm_stats。
路径：
- 输入：`outputs/sft_pi05_sponge/checkpoints/010000/pretrained_model/policy_preprocessor_step_2_normalizer_processor.safetensors`
- 输出：替换两个位置的 `norm_stats.json`（在 `~/.cache/huggingface/hub/.../pi05_base/.../aswinkumar99/...` 和 `/home/hlei/RLinf/assets/pi05_isaaclab_so101_lift/aswinkumar99/...`）

### R4（🟡 中）：lerobot vs openpi 参数命名差异 — 跨 codebase 加载 silent failure

**为什么是风险**：strict=False 加载允许 missing key 静默跳过。RLinf
之前用 HF `lerobot/pi05_base` 加载到 openpi 时观察到 811/815 keys
overlap，4 missing 是 buffers + tied weights。但 SFT 后 ckpt 的 key
集合**可能不一样**（lerobot 有 PEFT/projection 包装），需重新审计。

**应对**：Phase 1.5a：
- dump lerobot ckpt 的 keys
- dump RLinf openpi actor 期望的 keys（`get_model()` 入口）
- diff，输出 missing/unexpected/coverage % 报表
- 决定是写一个 key renaming hook，还是 ckpt 一次性 remap 后落盘

### R5（🟡 中-低）：Pi 0.5 flow-matching 的 PPO log_prob

**为什么是风险**：PPO 需要 `log π(a|s)`，但 Pi 0.5 是 flow-matching
模型，没有显式 log_prob。原 IROS plan 和当前 RLinf plan 都没解掉。

**已部分缓解**：RLinf 自带 openpi PPO worker 应该已经实现了某种近似
（ELBO / flow matching loss 的负值 / REINFORCE 退化等），但
**具体实现没核对过**。

**应对**：Phase 3 前读一次 RLinf 的 openpi PPO worker 源码，确认
log_prob 怎么算。如果是 ELBO 近似，留意 variance 大不大。

---

## 四、下一步执行顺序（建议）

按风险成本比排序。每步独立、可中止。

1. ~~**Phase 0**~~：已完成（2026-06-01），R2 降级为低。

2. **Phase 1.5a — key 审计**：dump lerobot ckpt keys + RLinf openpi
   expected keys，diff。1 小时。输出：`scripts/audit_ckpt_keys.py`
   + 结果报告。

3. **Phase 1.5b — normalizer 抽取**：从 lerobot
   `policy_preprocessor_step_2_normalizer_processor.safetensors` 抽
   state/action mean/std/q01/q99，写真值 `norm_stats.json` 替换之前的
   dummy。1 小时。输出：`scripts/extract_norm_stats.py`。

4. **Phase 1.5c — remap & 加载**：根据 1.5a 的 diff 写 key renaming
   hook（最小入侵）；在 RLinf openpi actor 加载入口接入。1-2 小时。

5. **Phase 2 — `only_eval: True` 验证**：fork PPO yaml，`model_path`
   指向 lerobot ckpt + 替换后的 norm_stats，跑 100 eval episodes。
   1-2 小时。**Decision gate**：
   - success rate ≥ 30% → 进 Phase 3
   - < 30% → 回到 R3/R4 排查（是否 key remap 错了 / norm_stats 算错了 /
     dataset action 单位真的不对）

6. **Phase 3 — PPO/GRPO 后训练**：开始前先解 R1（ViewBackward0），
   时间预估 1 天到 1 周（取决于走 patch / 降级 PyTorch / 替换 FSDP）。
   解掉之后 `only_eval: False` 启动 PPO，目标 ≥80%。12-24h 训练
   wallclock。

---

## 五、已完成（W1 详细）

- SO-101 URDF → USD 转换 + Isaac Sim 5.1 加载验证（`scripts/inspect_so101.py`）
- 两个 Isaac Lab 环境注册并跑通：
  - `Isaac-Lift-Cube-SO101-v0`：cube + 随机目标，RL playground
  - `Isaac-Lift-Sponge-Bowl-SO101-v0`：sponge + 固定 bowl 位姿，
    对齐 aswinkumar99 数据集
- 双相机 obs（`cam_high` 224×224 + `cam_wrist` 224×224 uint8）+ 6-DoF
  joint pos action 接口与 Pi 0.5 数据格式对齐
- PD 整定：`stiffness=1000, damping=50`（URDF importer 默认），
  零位姿能 hold，但 elbow_flex / wrist_flex 有 ~20° 稳态偏置（重力压塌，
  对 policy 推理不致命，对 RL critic reward 可能要观察）
- LeRobot SFT 通路（dry-run 主要用途变成产 ckpt）：`lerobot-train` 在
  `aswinkumar99/.../sponge-...` 数据集上跑 step 10000，loss 曲线健康，
  `bfloat16` + gradient checkpointing + train_expert_only=true 把 4B
  Pi 0.5 塞进 24GB 显存

---

## 六、变更历史

- **2026-06-02**：Phase 2 standalone eval 跑通链路，但 SR=0/N。**根因不是
  plumbing 也不是 norm_stats，是 cross-domain gap 三层 mismatch**：
  视觉（dataset 真实房间 vs sim CuboidCfg 海绵 + serving_bowl USD）+ 动力学
  （URDF 限位 ±100° vs dataset shoulder_lift mean=-103°）+ Train 域 0 sim
  数据。完整复盘见 `~/MemoryPalace/robotics/lerobot/phase2_eval_domain_gap.md`。
  **Pivot：放弃手搓 sponge-bowl env，接入 [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)**
  作为 git submodule（`third_party/leisaac/`），任务目标改为 leisaac 的
  `LeIsaac-SO101-LiftCube-v0`（自带配套 LeRobot dataset + 视觉对齐的 sim
  env）。旧 env 层（`src/lerobot_rlinf/tasks/lift/`、`assets/so101.py`、
  `assets/so_arm100/`、`third_party/IsaacLab/`、`scripts/cam_sweep.py` 等）
  全删；保留 Phase 1.5 ckpt remap 工具（任务无关，未来 lift_cube SFT 后复用）。
  IsaacLab 改用 leisaac 嵌套的 fork（2.3.0，downgrade from 0.54.3）。
  `scripts/smoke_lift_cube_leisaac.py` 跑通，obs/action 接口已确认。
- **2026-06-01**（晚）：跑 Phase 0 dataset replay，确认 action 单位
  对齐（stable mean err 5.69°，量级正确）。脚本阈值 FAIL 但根因是
  env 桌面高度截断 + wrist_flex PD 稳态偏置，与单位无关。**R2 风险
  降级为低**，不阻塞 Phase 1.5/2。
- **2026-06-01**：放弃 RLinf SFT 主路径，转向 lerobot ckpt + openpi
  remap。撤回所有 `openpi/models_pytorch/gemma_pytorch.py` 的临时
  patch。撤回 `use_orig_params: True`。新增本文件。
- **2026-05-31**：lerobot SFT 跑到 step 10000。同时尝试 RLinf SFT
  smoke，命中 R1 ViewBackward0 inplace 错误，多个 patch 均失败。
- **2026-05-30 / 31**：W1 env 完成并 commit；SFT 通路 install 链路打通
  （详见 `docs/notes.zh.md` "SFT 冒烟调试" 段）。
