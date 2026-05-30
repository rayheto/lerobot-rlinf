# lerobot-rlinf

**RLinf + Isaac Sim + LeRobot Pi 0.5 后训练框架**
针对 SO-ARM100 / SO-101 机械臂的仿真强化学习训练栈。

> English version: [README.md](README.md)

---

## 目标

在 Isaac Sim 中以 RLinf 为训练主循环，基于 LeRobot Pi 0.5 预训练权重（VLA 模型）做后训练（LoRA 微调为主），让策略在仿真环境中通过 PPO/GRPO 优化目标任务（首阶段：物体抓取）。

```
┌────────────────────────────────────────────────────────┐
│                  rlinf 训练主循环                       │
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

**核心原则：** LeRobot 只贡献模型结构和预训练权重，rlinf 完全控制训练循环、数据采集和梯度更新。

---

## 仓库结构（规划）

```
lerobot-rlinf/
├── README.md / README.zh.md
├── assets/                 # 机器人资产
│   └── so_arm100/          # URDF / USD / meshes
├── envs/                   # Isaac Sim 环境（OmniIsaacGymEnvs 兼容）
│   └── so101_pick_task.py
├── actors/                 # Pi 0.5 actor / critic wrapper
│   ├── pi05_actor.py
│   └── pi05_critic.py
├── trainers/               # rlinf 训练循环、PPO 适配
├── configs/                # YAML / Hydra 配置
├── scripts/                # 资产转换、对齐测试、训练入口脚本
└── third_party/            # 第三方仓库（submodule 或 clone）
```

---

## 技术栈与版本

| 组件 | 版本 | 说明 |
|------|------|------|
| Isaac Sim | 4.2.0 | Seeed Wiki 验证版本 |
| OmniIsaacGymEnvs | Isaac Sim 4.2 分支 | 向量化环境基础 |
| LeRobot | main | 提供 Pi 0.5 模型 |
| RLinf | main | 训练主循环框架 |
| PEFT | ≥ 0.10.0 | LoRA 实现 |
| PyTorch | ≥ 2.3.0 | bf16 autocast |

---

## 机器人：SO-ARM100 / SO-101

- 6 DOF（5 关节 + 1 夹爪）
- 资产来源：[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) → `Simulation/SO101/so101_new_calib.urdf`
- 在 Isaac Sim 中通过 URDF Importer 转 USD，固定底座，关节位置驱动

| 关节 | 名称 | 范围 |
|------|------|------|
| 0 | shoulder_pan | ±π |
| 1 | shoulder_lift | ±π/2 |
| 2 | elbow_flex | ±π |
| 3 | wrist_flex | ±π/2 |
| 4 | wrist_roll | ±π |
| 5 | gripper | [0,1] 归一化 |

---

## obs 协议（必须与 LeRobot Pi 0.5 训练集一致）

```python
obs_dict = {
    "observation.images.cam_high":  Tensor[B, 3, 224, 224],  # float32, [0,1]
    "observation.images.cam_wrist": Tensor[B, 3, 224, 224],
    "observation.state":            Tensor[B, 12],  # 6 joint pos + 6 joint vel
}
```

key 命名严格遵循 `observation.images.<cam_name>` 格式，否则 LeRobot 内部 embedding lookup 会出错。

---

## 后训练策略：LoRA

- **vision encoder 冻结**（SigLIP / PaliGemma backbone）
- **action expert 插入 LoRA**：rank 16~64，target `q_proj/k_proj/v_proj/o_proj`
- **flow matching log_prob** 是 PPO 接入核心难点，需确认 LeRobot 是否提供 `compute_log_prob()` 接口，否则用 `−flow_matching_loss` 近似
- **Critic 外挂**：Pi 0.5 没有 value head，使用独立轻量 MLP critic

---

## 里程碑

| 周次 | 目标 |
|------|------|
| W1 | 资产准备：URDF 导入 Isaac Sim，跑通无图像 `env.step()` |
| W2 | obs 格式对齐：双相机 GPU tensor 路径，Sim2Sim 对齐测试通过 |
| W3 | Rollout 数据流：Trajectory Buffer + Critic + eval 循环 |
| W4 | PPO 接入：log_prob 解决，首次完整训练 run |
| W5–6 | 稳定与扩展：dense reward、并行 env、domain randomization |

完整方案见 `/home/hlei/MemoryPalace/robotics/lerobot/rlinf_isaacsim_pi05.md`。

---

## 分支约定

- `main`：稳定版本，只接受 reviewed 的 PR
- `develop`：日常开发主分支
- `feature/*`：功能分支

---

## 关键风险

| 风险 | 应对 |
|------|------|
| Pi 0.5 无 `compute_log_prob()` | REINFORCE 先行；或 flow matching ELBO 近似 |
| 多 env 相机显存爆 | 单 env 起步，或降到 128×128 |
| URDF 物理参数不准 | 对比真机标定 `drive_strength` / `damping` |
| action chunk 与 single-step PPO 冲突 | 固定取 chunk[0]，后续考虑 multi-step PPO |

---

## License

待定（依赖组件分别遵循各自 license：LeRobot Apache-2.0，Isaac Sim 闭源 SDK，TheRobotStudio/SO-ARM100 见上游仓库）。
