# TODO

## SO-101 env (W2)

- [x] ~~Feetech 0% 标定锚点对齐~~。已废弃：env 改用 **degrees** 制
  （LeRobot v3.0 SO-101 数据集约定），0° 锚 URDF 机械零位，无需
  Feetech 锚点对齐。

- [ ] **PD 稳态误差**。stiffness=1000 在 elbow_flex / wrist_flex 上零位
  hold 仍有 ~20° 稳态误差（重力压塌）。对 Pi 0.5 推理不致命（policy
  会发非零 hold），但 W3 跑 PPO 时 critic 看 reward 可能受影响。届时
  考虑：
  - 把 stiffness 拉到 2000 看是否收敛
  - 或加 actuator-level gravity compensation
  - 或对 reward 做 vel-based shaping，不依赖绝对姿态

## SFT path (无实机方案)

走 Path A：直接用 aswinkumar99 公开数据集做 SFT，跳过真机遥操采集。

- [ ] 选数据集（task1/2/3 × random/fixed，各 64 episodes，共 7 个公开
  数据集，无 gating）。task1-random 已 verified。
- [ ] 写 SFT 训练脚本：`lerobot-train` finetune `pi05_base` on chosen
  dataset。
- [ ] 写 `LeRobotPi05Wrapper`：env 输出 → `observation.state` /
  `observation.images.{front,wrist}` 键名重命名；image uint8
  [B,H,W,3] → float [0,1]；action chunk `[B,50,6]` 按时间步索引。
  action/state 单位已经是 degrees，对齐数据集 → 不需要单位转换。
- [ ] 在 sim 里部署 finetuned ckpt 验证 pick-and-place。
- [ ] RLinf 后训练：用 SFT ckpt 当 init policy 跑 PPO。

## env state ↔ 数据集 state 对齐（仅当需要 env-state 校验时）

`scripts/replay_dataset_actions.py` 已经确认动作单位对齐（度、URDF 顺序，
stable mean ≈ 5°）。但**关节级跟踪精度不一致**，按 ep0 数据：

- shoulder_pan: 1.5° ✓
- shoulder_lift: 4.0° ✓
- elbow_flex: 6.2° 边缘
- **wrist_flex: 11.5°** 差（PD 跟不上 / 速度限幅截断 / 重力压塌）
- wrist_roll: 4.3° ✓
- gripper: 7.7° 边缘

**对 SFT 不致命**：SFT 学的是数据集图像 + 动作，不依赖 env state 和数据集
state 对齐。
**对 RL eval 也不致命**：用 success-rate 当指标，不看 env state 误差。
**只有以下场景要修**：用 env state 当 ground-truth 反过来评估数据集 /
policy 输出，或 env-state-based reward shaping。

**根因（2026-05-31 诊断）**：不是 PD 也不是单位 —— 主要是**物理约束错配**。
数据集采集场景的桌面更低（或没桌面），允许更大动作空间；我们 env 里
表面高度截断了部分轨迹，机械臂触底卡 1-2 step。表现：t=250 / t=500
两个 mean 尖峰（21° / 32°），不是持续漂移。wrist_flex 11° 那个稳态
偏置才是 PD/重力问题。Sim2real 阶段才需要彻底对齐 workspace。

- [ ] 单独把 wrist_flex 的 stiffness 拉高（其他 5 个关节维持 1000，避免
  影响已 verified 的整体行为），或 reward 设计绕开姿态绝对值。
- [ ] 同时排查 elbow_flex / gripper 边缘 case 是不是同源（重力 + 速度限）。

## 未确认事项

- [ ] **物体 USD**：env 里 spawn 的是 DexCube，aswinkumar99 数据集任务
  可能是别的物体（task1/2/3 具体内容待确认）。如果数据集是 cube
  pick-and-place 直接用；否则需要换 mesh 或验证 cube 当代理物体是
  否可用。
