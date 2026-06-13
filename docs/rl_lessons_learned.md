# RL 后训练实践：经验与教训

> 一份把四份过程文档（Step 5 / JAX-24999 dryrun / Step 8f 交叉分析 / Step 9 报告）合
> 并后的回顾。按"出过的坑 → 根因 → 解法 → 学到了什么"组织。
>
> 范围：`src/rl/simple/` 单进程 PPO 残差头，在 SO-101 pick_orange 上对冻结 π₀.₅ +
> LoRA 的 SFT ckpt 做后训练。最终落点：SFT 56.67% → RL iter100 72.50%（90s 严格
> 预算，n=60）。
>
> 原始材料：
> - [docs/simple_ppo_step5_report.md](simple_ppo_step5_report.md)（1k env step 框架冒烟）
> - [docs/dryrun_jax_24999_diagnostic.md](dryrun_jax_24999_diagnostic.md)（SFT 精度上限诊断）
> - [docs/dryrun_jax_step8f_crossanalysis.md](dryrun_jax_step8f_crossanalysis.md)（50-iter 训练 + 三方 eval 交叉分析）
> - [docs/simple_ppo_step9_report.md](simple_ppo_step9_report.md)（200-iter 最终训练）

---

## 一、架构基线 — 两个 head 必须分清

整套 RL **只训一个非常小的 PyTorch 残差头 + value head**，π₀.₅ 主干 + LoRA 全程冻结：

| 名称 | 在哪 | 内容 | 大小 | RL 阶段 |
|---|---|---|---|---|
| **π₀.₅ action expert / flow head** | `outputs/.../24999/`（JAX server 加载） | π₀.₅ 主干 + flow-matching action expert + LoRA | ~6 GB | **冻结** |
| **residual head + value head** | `head_*.pt`（PyTorch） | `MLP(12→256→12)` 输出 (μ, log σ)；`MLP(12→256→1)` 输出 V(s) | ~MB | **唯一在训练的部分** |

`action = base + r`，其中 `base` 是 π₀.₅ 给出的 chunk slot，`r ~ N(μ, σ²)` 由残差头采。
**所有 RL 故障/收益都是这个小 MLP 在动**，主干一字未改。这条边界后面每一节都要用到。

---

## 二、问题与根因 — 按发现顺序

### 2.1 稀疏 reward 在 SFT 起点上是 0 信号

**症状（Step 6 早期）**：31 episodes 全部 timeout，`mean_env_reward ≡ 0`，PPO 内部健康
（KL 正常、grad bounded），但**没有任何可学的信号**。

**根因**（[dryrun_jax_24999_diagnostic.md](dryrun_jax_24999_diagnostic.md)）：把 JAX
ckpt 24999 起 serve 然后跑 dryrun，900 步全程 trace 三个谓词量：

```python
is_grasped = (d_ee_orange < 0.05) & (gripper < 0.60) & (lift_dz > 0.06)
```

- `d_ee_orange` 全程 0.07–0.10 m，**最近 0.070 m，永远跨不过 5 cm 阈值**
- `gripper` 频繁满足 < 0.60（这一项不卡）
- `lift_dz` 恒 ≈ -0.006 m（橘子根本没被夹起，物理 settle 噪声）

→ 末端定位精度差 2–5 cm，sparse 3 阶段谓词永远不触发 → grasp_bonus / place_bonus / drop_penalty
全是 0 → PPO 只剩 `timeout_penalty=-2`，advantage 几乎是常数。

**学到的事**：
- **sparse reward 不是"环境本来就有的"信号——它是 SFT 能否触发某些边界的副产品**。
  在末端精度差最后一公里的策略上，sparse reward 就是零。
- 诊断方法：**不要看 PPO 健康指标判断"训练有没有信号"**，那只能告诉你优化算法本身没坏。
  要直接探针环境谓词的命中率。
- 这个差最后一公里**正是 RL 要修的精度问题，不是 ckpt / wrapper 的 bug**——后两者一开始都被怀疑过。

**解法**：加 dense shaping 给一个连续梯度。具体见 2.2 / 2.5。

---

### 2.2 dense_eo 是状态相关的稳态 bias，不是"系数太大"

**症状（Step 8f）**：50 iter 训完，eval iter50 ckpt 17.5%，**比 zero-residual 基线 28.3%
还低**——RL 把性能训差了。

**根因**（[dryrun_jax_step8f_crossanalysis.md](dryrun_jax_step8f_crossanalysis.md) §2.1, §6.1）：

最初为了给末端定位精度差一公里加梯度，引入了 `dense_eo = -1.0 · max(0, d_ee_orange - 0.05)`。
按 SFT 实际 `d_eo ≈ 0.136 m` 反推，**每 episode 累计 −77 / -59**，而 env_reward 50 iter 才到 +47.89：

| 分量 | iter 0 | iter 49 |
|---|---|---|
| env_reward | +0.51 | +47.89 |
| **dense_eo** | **−77.25** | **−59.00** |
| dense_lift | 0 | +7.67 |
| survival_cost | −0.90 | −0.90 |
| **shaped_reward (PPO 实际优化的)** | **−77.64** | **−4.34** |

iter 0 一次更新 **`approx_kl=4.47`（target 的 90×）、grad_norm=325**——一步就把 head 推
出 trust region。机制：

- iter 0 的 value head 零初始化，V(s) ≈ 0。
- 第一批 rollout return ≈ env + dense_eo ≈ 0 + (−77) = −77。
- 整个 batch advantage = R − V ≈ −77，**所有样本同号、量级巨大**。
- PPO 把残差 μ 推到任何能让 d_eo 减小的方向，把 σ 顶到 `log_std_min=-2.5` 下沿。

后续 50 iter 都在追一个**被 dense_eo 主导 (−60)、env_reward 永远追不上 (+48)** 的负
shaped target，策略学会的是"伸直手臂把末端推近橘子"的退化解——**与 SFT 良好状态分布
背道而驰**，恰好复现 [sft_diagnostics_findings.md](sft_diagnostics_findings.md) 的 OOD 低速 attractor。

**学到的事**：

- **大幅度、状态相关、单方向的稠密项 ≈ 在 reward 里注入稳态 bias**。改 clip / lr /
  warmup 都治不了根，因为 PPO 优化的目标本身就在告诉策略"远离当前状态"。
- **`grasp_bonus` 已经把"抓到"编码成 +10 脉冲**了。再加 dense_eo = "未抓取时持续扣分"
  ——双计且方向不一致（一个事件触发、一个连续势能），等于和原任务梯度对抗。
- **单变量代理 = 退化路径**：dense_eo 只看末端到 orange001 的距离，策略可以靠"伸长
  手臂"减少 d_eo，但姿态 OOD、夹爪方向错——拿不到 grasp。
- **正确做法是 potential-based reward shaping**（Ng99）：
  `Φ(s) = -d_eo`，增量 `γ·Φ(s') − Φ(s)`，期望值为 0 → 不引入 bias，仍提供"靠近 orange"
  的梯度。本仓库留作 Step 10 接口草案未实现。

**解法**（Step 9）：直接把 `dense_eo_coef` 从 1.0 → 0.0，只留 dense_lift（橘子静止时该项=0，
没有稳态偏置）+ 微弱 survival_cost。结果见第三节。

---

### 2.3 BC anchor 在 sparse-zero 环境里把熵抽干

**症状（Step 5）**：1024 env step 跑完，**熵从 2.09 塌到 −0.60，对应 σ ≈ 0.10**，
exploration 几近消失。

**根因**（[simple_ppo_step5_report.md](simple_ppo_step5_report.md) §4.2）：

- `r^env ≡ 0`（见 2.1）→ advantage 没方向信号。
- BC anchor `L^bc = -log π(r=0 | s, b=a_demo)` 全程拉 σ → 0（demo 上 r=0 最优）。
- `ent_coef = 0.003` 偏弱，挡不住 BC 压熵。
- `log_std_min = -5.0` 给的下沿太低，触底也才 σ ≈ 0.007——形同没限。

→ 10k+ iter 后 policy 退化为 deterministic = pi05 base，等同 SFT-only，RL 没学到。

**学到的事**：

- **BC anchor 的方向（让残差→0）和 探索目标（保持 σ）天然冲突**。当 env_reward 给不出
  正信号时，BC 项是 reward 里唯一"有方向"的力，会单边把熵抽干。
- 不能依赖"加大 ent_coef"硬抗 BC——量级很难调对。**最简洁的修法是 σ 硬下限**（log_std_min）
  把 σ 锁在探索可用的最小值，让 BC 压不下去。

**Step 6 前的修复**：
- `log_std_min: -5.0 → -2.5`（σ ≥ 0.08）
- `ent_coef: 0.003 → 0.01`（最终 Step 9 又调回 0.003，因为 BC 也关掉了）
- rollout_len 64 → 256，total_iters ≥ 200

**Step 9 终局**：BC anchor 整个 `bc_coef = 0`，因为后来发现 SFT 起点已经在 demo manifold
内，BC 项是死重——见 2.4。

---

### 2.4 OOD penalty 在 SFT 起点上是"惩罚正确行为"

**症状（Step 9 前）**：开启 `ood_coef > 0` 时，每 episode 累计 OOD 惩罚 −45 ~ −75，**直接
压制 grasp_bonus(+10)**。

**根因**：`OodKNNPenalty` 在 60 条示教里建关节空间 KD-tree，每步算 5-NN 距离归一化为
`d_norm`。SFT 起点已经基本贴在 demo manifold 上，`d_norm` 大部分时刻在 1 附近——这种状态
下加 OOD 惩罚等于**在策略本来就做对的地方扣分**。

**学到的事**：

- **OOD penalty 是"修离开示教流形"的工具，不是预防措施**。诊断里 `coverage_ratio = 5.07`
  是 SFT 已经训完之后跑 eval 时观察到的 OOD 行为，不是 SFT 起点时的状态。
- 给 SFT 起点加 OOD 惩罚 = 把 reward 几何拽偏成"远离当前最优"——和 2.2 dense_eo
  同型的错误。

**解法**：`ood_coef = 0.0`，代码保留备日后用。

---

### 2.5 Eval pipeline 不一致放大 ΔP

**症状（Step 8f）**：同一份 ckpt（24999），src/eval.py 报 ~60%，simple/eval.py 报 28.3%
（zero-head）——**32 pp 缺口看起来像 ckpt 坏了**。

**根因**（[dryrun_jax_step8f_crossanalysis.md](dryrun_jax_step8f_crossanalysis.md) §3, §6.2）：四处不一致：

| 字段 | src/eval.py（参考） | simple/eval.py（旧） | 影响 |
|---|---|---|---|
| sim 频率 | 30 Hz outer | 60 Hz outer | **2× 加速**：pi05 chunk 表示 333 ms 的轨迹，被 167 ms 内播完→关节速度命令 2× → 轨迹被压扁 |
| `episode_length_s` | **90 s**（max ep len 2700@30fps 反推得） | 45 s | 长尾里的成功被裁掉 |
| success 口径 | leisaac native | 3 orange + rest 严判 | 严了 |
| prompt 字符串 | 训练用 | 不一致 | 微小但有 |

更绕的是：**原文档"60% baseline"用的实际是 90s 预算**——一开始误以为是 src/eval.py CLI
默认值 60s，导致前期 simple/eval.py 长期用 45s 比较，把 SFT v3 56.67% 算成 28.33%，
把所有 RL 结果都低估约 20 pp。直到对 `outputs/.../24999/dataset/meta/episodes.jsonl` 反查
最大 ep 长度才发现真相。

**学到的事**：

- **任何"我的 RL 把性能训坏了"结论之前，先确认 eval pipeline 与基线 apples-to-apples**。
  Step 8f 32 pp 缺口里有 32 pp 是 eval-infra，0 pp 是 RL——和最初的误判正好相反。
- **物理时间 = sim_dt × decimation × outer_step**，三者必须和训练数据保持一致。SFT 数据
  在 30 fps 录制 → outer 必须 30 Hz → `sim.dt=1/60, decimation=2`。
- **官方报数的真实参数要直接从落盘的 meta 反查**，不要假设 CLI default。
- **零残差头 ckpt (`head_zero_init.pt`) = 通过同一份 eval 流水线测 SFT 基线的方法**。这是
  本仓库长期采用的 apples-to-apples 基线机制。

**解法**：Step 9 在 env wrapper 端 `decimation 1 → 2`；eval CLI 加 `--episode-length-s
--max-ep-steps` 直接覆盖。最终 v3 评测严格对齐 90 s 预算。

---

### 2.6 cleanup hang（multiprocessing.resource_tracker）

**症状**：train/eval 主进程 SUMMARY 打出来后卡在 `futex_wait`，`kill -0` 监听不到退出。

**根因**：multiprocessing.resource_tracker 子进程在 atexit 里互锁。ckpt 在死锁前已经
落盘所以**不影响结果**，但 orchestrator 串行 pipeline 必须有 watchdog。

**解法（短期）**：orchestrator 在 stdout 里 grep `SUMMARY`，命中后直接 `kill -9` 主进程。
脚本：`/tmp/eval_pipeline_v3.sh`。

**解法（永久，未做）**：`train.py` 和 `eval.py` 退出路径加 `os._exit(0)` 绕过 atexit。

---

## 三、最终配置与结果（Step 9, 200-iter）

### Reward 配置（核心改动）

| 参数 | 值 | 说明 |
|---|---|---|
| `dense_eo_coef` | **0.0** | 关掉单变量距离代理，避免 2.2 描述的 bias |
| `dense_lift_coef` | **2.0** | 只在 orange 物理离地（z > z0）才计 + 2·Δlift，无稳态偏置 |
| `ood_coef` | **0.0** | SFT 起点已在 demo manifold 内，OOD 惩罚反而压制正信号 |
| `survival_cost` | **−0.001** | 2700 步累计 −2.7，与 grasp_bonus +10 量级一致，给轻微时间压力 |
| `bc_coef_*` | **0.0** | BC anchor 在 SFT 起点是死重，删 |
| `ent_coef` | 0.003 | log_std_min=-2.5 已经给 σ floor，无需高 ent_coef 抗压 |
| `lr` | **1e-4** | 3e-4 → 1e-4，因 iter-0 KL 仍偏大 |
| `total_iters` | 200 | `env_num_envs=8, rollout_len=64 → batch=512/iter` |

环境侧 sparse reward 不变：grasp_bonus=10 / carry=0.5·|Δee| / place=20 / drop=−5 / timeout=−2 / rest_bonus=30。

### 评测对比

| ckpt | 45 s 预算 succ | 90 s 预算 succ | 90 s fast<900 |
|---|---|---|---|
| **SFT baseline**（零残差头） | 28.33% | **56.67%** | 28.33% |
| **RL iter100**（90s 取均） | 53.33% | **72.50%**（avg 73.33/71.67） | 43.33% |
| **RL iter200** | 61.67% | 68.33% | 36.67% |

- 文档原 SFT eval_run.py 报 60.0%；SFT v3 56.67% 偏差 3.3 pp，在 n=60 的 1 σ ≈ 6 pp 内
  → **严格同口径复刻成立**。
- 90 s 预算下 RL 净提升 iter100 +15.83 pp，iter200 +11.66 pp。
- iter100 > iter200 翻转（45 s 下是反过来）：~4 pp 差落在噪声边缘，需 ≥3 seed 才能判定
  是 iter200 略 overfit 长尾还是 seed 噪声。

完整报告：[docs/simple_ppo_step9_report.md](simple_ppo_step9_report.md)。

---

## 四、跨阶段沉淀下来的方法论

### 4.1 "哪个 head 在动" 是任何故障分析的第一问

整条 stack 里只有那个小 MLP 在更新，pi05 主干和 LoRA 都冻结。这意味着：
- 任何性能变化必定是残差 head + value head 的训练动力学造成的。
- 评测一份 head ckpt 时，可以用 `head_zero_init.pt`（μ ≡ 0）跑同一份 eval 拿"等价 SFT"
  基线——这是本仓库长期采用的 apples-to-apples 模式。

### 4.2 Reward 设计的三个判据

从 dense_eo 翻车里提炼：

1. **零基线稳定性**：在 baseline 策略上，该项的期望累计值是不是 0？dense_lift 是（橘子
   不动时 lift=0），dense_eo 不是（SFT 末端总在 7 cm 远 → 每步 −0.086 → 每 ep −77）。
2. **与 sparse 子项的非重叠**：grasp_bonus 已经编码"抓到"，再加 dense_eo 是双计且方向不一致。
3. **单变量代理 vs 多变量任务**：dense_eo 只看 d_ee_orange 一维，策略可以靠"伸长手臂"
   减少它而其他维度全错——避免单变量代理，或退到 potential-based 形式（Ng99）让 bias 严格为 0。

### 4.3 eval pipeline 要先对齐再下结论

Step 8f 的 32 pp"RL 退化"几乎全是 eval-infra 不一致造成的。规矩：

- 任何"RL 比 SFT 差"结论之前，必须先跑 `head_zero_init.pt` 走**同一份 eval 流水线**拿 SFT
  基线。
- 物理时间 = `sim_dt × decimation × outer_step`，所有数字必须和训练数据匹配。
- 官方 baseline 的真实参数从落盘 meta 反查，不依赖 CLI default。

### 4.4 调试 sparse reward 不依靠 PPO 内部指标

PPO 健康（KL 正常、grad bounded）只说明优化器没炸，不说明环境给了信号。要直接 trace
环境谓词的命中率（`d_ee_orange`, `gripper`, `lift_dz`）。Step 6 的 31 episodes / 0 success
事故就是被 PPO "看起来健康"误导了一阵。

### 4.5 σ 用硬下限，不靠 ent_coef 抗 BC

Step 5 熵塌缩教训：BC anchor 是单边压熵的力，ent_coef 调到能抗它的量级会把策略损失淹没。
正确做法是 `log_std_min` 直接给 σ floor（实际取 -2.5 → σ ≥ 0.08），让 BC 压不下去。Step 9
连 BC 都关了，但 σ floor 保留作为防退化兜底。

---

## 五、未决项

- **多 seed 统计**：当前 N=1，iter100 / iter200 谁好 4 pp 在噪声里看不清，要 ≥3 seed。
- **训练饱和点**：iter100 → iter200 在 45s 下还在涨，90s 翻转——300/400 iter 看天花板。
- **potential-based dense_eo**：Ng99 形式的 `Φ(s) = -d_eo`，理论 bias 为 0，预期能在不
  退化的前提下加速末端精度收敛。`ShapedReward` 里需要维护 `phi_prev` 并处理 reset。
- **cleanup hang 永久修**：`train.py` / `eval.py` 退出路径 `os._exit(0)`。
- **真实机器**：所有结论目前只在仿真里站得住。
