# SFT 诊断结果与问题归纳 — pi05-LoRA ckpt 24999

- 任务：`pick_orange`（SO-101 / leisaac）
- 训练框架：openpi pi05 + LoRA + LeRobot v2.1 数据集（60 条专家示教）
- 待测落盘：`outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999/dataset/`
- 参考集：`/home/hlei/.cache/huggingface/lerobot/EverNorif/leisaac-pick-orange`
- 诊断框架：`src/diagnostics`（5 模块全跑通；commit `8ab4710`）

## 1. 观测到的现象

| 维度 | 实测 |
|------|------|
| 成功率 | **60–70%** |
| 时长劣化 | 约 2–3×（实测 episode 帧数 605 → 1654 @ 30fps，即 20.2s → 55.1s） |

## 2. 五模块诊断量化结果

| Module | Status | 关键指标 | 简释 |
|--------|--------|---------|------|
| EXP_01 Mode Averaging       | WARNING  | `ratio = 0.726`           | 动作分布散度轻度塌缩（mode covering）|
| EXP_02 Compounding Error    | OK       | `path_len_ratio = 1.262`  | 关节空间累计弧长基本正常 |
| **EXP_03 Episode-Length Inflation** | **CRITICAL** | `length_ratio = 2.735` | 时长劣化 2.7×（症状直测，命中观测）|
| EXP_04 Action Smoothness    | OK       | `smoothness_ratio = 1.298`| 无显著 chunking / EMA 伪迹 |
| **EXP_05 State Coverage Divergence** | **CRITICAL** | `coverage_ratio = 5.07` | 关节构型严重偏离示教流形 |

## 3. 当前训练存在的问题（按诊断证据归类）

### 3.1 主问题：策略在 OOD 关节构型中长时间停留

- **直接证据**：EXP_05 `coverage_ratio = 5.07`——候选 episode 里的 6 维关节构型，
  到示教集合的最近邻距离是示教内部尺度的 5× 以上。
- **后果证据**：EXP_03 `length_ratio = 2.735`——平均 episode 时长从 20.2s 膨胀到 55.1s。
- **机理判读**（EXP_02 OK + EXP_05 CRITICAL 的对照）：
  - EXP_02 `path_len_ratio = 1.262` 表明**每帧关节移动量没显著增大**——策略并非"绕远路"。
  - 但 EXP_05 表明这些关节构型本身不在示教分布内。
  - 结合 EXP_03 的时长膨胀 → 唯一一致的解释是策略**陷入 OOD 关节区域的低速 attractor**，
    每帧动得不远，但**待得很久**。
- **典型场景假设**（无图像信息时只能给候选）：
  抓取失败 → 反复接近-后撤；末端定位偏差 → 在错误高度长时间下探；陷入中间姿态 attractor。

### 3.2 次问题：轻度 mode covering，但非主因

- EXP_01 `ratio = 0.726`（WARNING 上沿）表明动作分布散度有压缩，与 L2 BC 目标下的
  mode covering 一致，但量级远不足以独立解释 2.7× 时长劣化：
  ```
  (1/EXP_01.ratio) × EXP_02.path_len_ratio = 1.377 × 1.262 ≈ 1.738
  EXP_03.length_ratio                                     ≈ 2.735
  ```
  动作侧+弧长侧只能解释约 64% 的时长膨胀（1.738 / 2.735），**剩余 36%** 主要来自
  EXP_05 揭示的 OOD 停留。

### 3.3 已被排除的假设

| 假设 | 反例证据 | 结论 |
|------|---------|------|
| Compounding error → 路径膨胀 | EXP_02 OK | **排除**为主因 |
| Action chunking / EMA 压平动作 | EXP_04 OK | **排除** |
| 策略高频抖动 | EXP_04 OK（ratio=1.298，未触 WARNING） | **排除** |

## 4. 诊断盲区（当前框架无法独立验证）

按"只读 LeRobot 数据集，不跑模型/不读图像"的解耦约束，以下来源未被覆盖：

- **推理延迟（wallclock）**：若 SFT 部署回路里推理耗时增加，会反映为 fps 下降。
  EXP_03 的帧数比 2.735 与时长比 2.735 一致**仅当两侧实际 fps 相同**。
  `meta/info.json.fps` 两侧均声明 30，但这只是 metadata，不保证部署 wallclock。
- **物体位姿 OOD**：橘子 / 盘子的位置变化未进入 `observation.state`，
  只在图像里。若需直接量化"橘子被碰飞"这类场景，需在 candidate 数据集补
  `observation.object_pose` 列，注册 `EXP_06_Object_Coverage_Divergence`（增量成本 O(1)）。
- **失败原因分布**：当前指标按全部 episode 聚合，未区分成功 / 失败子集。
  分箱后看 EXP_05 在两组上的差异，可判断 OOD 停留是失败专属还是普遍现象。

## 5. 复现命令

```bash
python -m src.diagnostics \
  --ref /home/hlei/.cache/huggingface/lerobot/EverNorif/leisaac-pick-orange \
  --cand outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999/dataset \
  --meta-json /tmp/diag_real_meta.json \
  --out-json /tmp/diag_real.json \
  --out-md   /tmp/diag_real.md
```

参考报告：[/tmp/diag_real.md](/tmp/diag_real.md)、[/tmp/diag_real.json](/tmp/diag_real.json)。

## 6. 一句话结论

> 当前训练的主要问题**不是**动作幅度被 mode covering 压扁，**也不是**轨迹被
> compounding error 拉长，而是**策略陷入了示教未覆盖的关节构型区域并在其中
> 低速磨蹭**——这是 EXP_03（症状）+ EXP_02 OK + EXP_05 CRITICAL 三者联读后唯一
> 自洽的解释。
