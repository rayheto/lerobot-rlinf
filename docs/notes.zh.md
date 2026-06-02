# 踩坑笔记

把 SO-101 接到 Isaac Sim 5.1 + Isaac Lab 过程中遇到的问题汇总。
留个底，省得下次再花一遍时间。

## SO-101 本体

- **零位姿就是站立位姿**。和 Franka 反直觉（Franka 的 home 是一组非零关节角）。
  用 `inspect_so101.py --pose` 验证过，`0 0 0 0 0 0` 就是合理 ready-pose。
  抄 Franka 经验填非零初值（比如 `shoulder_lift=-0.6, elbow_flex=1.0`）反而把臂折塌了。
- **关节顺序**：`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`（6 DoF）。
  command/reward 用的末端 body 名 = `gripper_frame_link`。
- **关节限位**（按上面顺序，来自 URDF parse）：
  下限 `[-1.920, -1.745, -1.690, -1.658, -2.744, -0.175]`，
  上限 `[+1.920, +1.745, +1.690, +1.658, +2.841, +1.745]`。
- **工作半径 ~20 cm**，Franka 是 ~80 cm。从 Franka 移植任务时
  command/spawn 范围都要按比例缩，不然 cube/目标点根本够不着。

## PD 整定

- Franka 的 actuator 增益（`stiffness=80, damping=4`）对 SO-101 **太软**——
  零位姿下臂自重就压塌了。
- URDF importer 默认值（`stiffness=1000, damping=50`）能扛住，
  现在 `SO101_CFG.actuators` 用的就是这一套。
- **为什么 `inspect_so101.py` 看不出这个问题**：它用 `robot.set_joint_positions()`，
  这个 API 是直接 teleport（同时写位置和目标），绕过 PD 回路。
  Lift env 走 `JointPositionActionCfg` → PD 控制器，才会暴露 PD 太软的事实。
  **PD 必须用零动作循环或真 policy 测**，不能用 teleport 测。

## Isaac Sim 5.1 / Isaac Lab 坑

- **从 4.x 改名了的 API**：`omni.isaac.core` → `isaacsim.core.api`；
  `Articulation` 类挪到 `isaacsim.core.prims.SingleArticulation`；
  `SingleArticulation` 没有 `get_joint_limits()`（限位在 PhysX dof properties view 里，
  或者干脆从 URDF parse 输出里硬编码）。
- **`SimulationApp` 启动前 `pxr` 不可用**。任何在 `AppLauncher`/`SimulationApp`
  上下文之外 `import isaaclab` 的测试都会在 `from pxr import Usd` 处报错。
  import 测试必须放进 SimulationApp 里。
- **`gym.make("Isaac-...")` 收的是 cfg dataclass，不是 dict**。
  得先 `isaaclab_tasks.utils.parse_env_cfg(task, device=..., num_envs=...)`。
- **argparse 和 SimulationApp 抢 argv**。SimulationApp 会把剩余 `sys.argv`
  转给 Kit，Kit 不认我们的 flag。init 之前要先剥掉：`sys.argv = sys.argv[:1]`。
- **Kit 会吞掉后期 stdout**。`python -u` + `flush=True`（用
  `functools.partial(print, flush=True)`）都加上，否则会丢日志行，看着像静默退出。
- **`omni.ui.Window` 在远程 DISPLAY 下显示不稳**。`position_x/y`、
  `dockPreference`、`window.visible=True` 都试过——`DISPLAY=:110` 下都没法稳定
  把窗口浮出来。改用工作线程 + stdin REPL 方案（参考 `inspect_so101.py --pose`），
  不要硬刚 omni.ui。

## 安装的小坑

- `pip install -e source/isaaclab*`（5 个包）会把 `psutil` 升到 >=7，
  但 `isaacsim-kernel==5.1.0` 死锁 `psutil==5.9.8`。
  **装完要补一刀**：`pip install "psutil==5.9.8"`（ipython 会抱怨但能跑）。
- `fastapi 0.115` 要 `starlette<0.46`，isaaclab 拉的是 0.49——忽略，
  fastapi 不在训练/rollout 链路里。
- IsaacLab 是 editable 装在 `third_party/IsaacLab/`（已 gitignore）。
  `git pull` 之后重跑 `pip install -e source/*` 就行，不用再跑 `isaaclab.sh -i`。
- **`rlinf-isaacsim-env` 里不要装 `lerobot`（或 `accelerate`）。**
  这俩会拖 numpy>=2 进来，Isaac Sim 5.1 直接崩 —— 几百个
  `isaacsim.*` 扩展全报 `AttributeError: module 'numpy' has no
  attribute '_no_nep50_warning'` 加载失败。`isaacsim-kernel==5.1.0.0`
  锁的是 `numpy==1.26.0`。如果要在 Isaac env 里读数据集（比如
  replay sanity 脚本），直接拿 pyarrow 读 `HF_LEROBOT_HOME` 下的
  parquet 分片，不要 import `LeRobotDataset`。已经踩坑了的修法：
  `pip install "numpy==1.26.0"`，下次跑 Isaac 扩展会重新 load 上。

## 可以安全忽略的警告

- `Unresolved reference prim path .../visuals/gripper_frame_link`——
  URDF importer 没给这个 fixed 末端坐标系写视觉子层（它本来就没 mesh）。
  物理正常，单纯没贴图。
- `getAttributeCount / getTypes called on non-existent path
  .../wrist_link/visuals/wrist_roll_pitch_so101_v2/node_STL_BINARY_`——同一族问题，
  STL 转换残留路径。
- `omni.isaac.dynamic_control` / `omni.isaac.wheeled_robots` 的 deprecation 警告——
  IsaacLab 里的三方扩展在喊，不是我们的代码。
- 首次 import Isaac Sim 5.1 弹 **EULA**：用 `echo "Yes" | python ...` 一次就过
  （注意：`DISPLAY=:110 echo Yes | python` 是错的，环境变量绑到 echo 上了，
  要写成 `echo Yes | DISPLAY=:110 python ...`）。

## LeRobot Pi 0.5 接口约定

2026-05-30 从 `huggingface/lerobot` main 分支和 `lerobot/pi05_base` 模型卡
拉的事实，记一下省得 W2 接的时候再查一遍。

- **Obs key 常量**（`src/lerobot/utils/constants.py`）：
  `OBS_STATE = "observation.state"`，`OBS_IMAGES = "observation.images"`
  （前缀，完整 key 是 `observation.images.<camera_name>`），
  `ACTION = "action"`。
- **相机名是数据集决定的，不是 policy 钉死的**。`pi05_base` 里挂的
  `observation.images.{base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}`
  只是例子；SO-101 finetune 用自己的（通常 `front` + `wrist`）。
  finetune 数据集用啥 key，policy 就吃啥 key。
- **图像格式**（`modeling_pi0.py`）：`[B,C,H,W]` 和 `[B,H,W,C]` 都接，
  不用我们手动 permute。数值要 float 范围 `[0,1]`。
  policy 内部 resize 到 `(224, 224)`，再归一化到 `[-1,1]` 喂 SigLIP。
- **Action 输出**：chunk `[B, chunk_size=50, max_action_dim=32]`
  （pad 到 32，真实 action_dim 由 finetune 决定，SO-101 = 6）。
  Pi 0.5 用 `NormalizationMode.MEAN_STD` 归一化 STATE 和 ACTION；
  `select_action()` 出来之前会反归一化回**数据集单位**。
- **Action 单位不是弧度**。是 SO-101 数据集采的单位，
  对 LeRobot Feetech 校准过的遥操数据集来说是归一化的角度类电机值。
  转成 env 关节目标必须读数据集的 action stats，**不要假设是弧度**。

**对 env 的含义**：以上这些都不要 hardcode 进 env cfg。
key rename、dtype/scale、数据集 → env action 单位换算这些事
全部放到 runtime wrapper 里写，等拿到具体的 finetune 再动手。

## SFT 冒烟调试（Pi 0.5 + lerobot 0.4.4）

把 `scripts/sft_smoke.sh` 跑通的踩坑记录，省下次几个小时。环境为
`rlinf-lerobot-train`。

- **`lerobot[pi]` extra 在 0.4.4 里是坏的。** pip 会警告 extra 不存在；
  基础安装能成功但缺 `transformers`。需要手动装，而且不能装 vanilla。
- **Pi 0.5 需要打过补丁的 transformers fork。** vanilla `transformers`
  里没有 `transformers.models.siglip.check` 模块，`modeling_pi05.py`
  加载时直接 ImportError。装 lerobot-openpi 分支：
  ```
  pip install --force-reinstall --no-deps \
    "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
  ```
  解析为 4.53.3。`--no-deps` 保住 hf-hub<0.36（lerobot 的 pin）。
  验证：`from transformers.models.siglip import check;
  check.check_whether_transformers_replace_is_installed_correctly()` 返回 True。
- **第一次 `pip install ... git+...` 可能静默跳过。** pip 看到缓存版本
  就不装了，第一次必须加 `--force-reinstall`，不是可选项。
- **数据集 revision pin。** 公开 LeRobot 数据集常常没有代码侧检查的
  v3.0 git tag → `RevisionNotFoundError`。CLI 加 `--dataset.revision=main` 绕过。
- **PaliGemma tokenizer 是按 HF 账号 gate 的。** Pi 0.5 会拉
  `google/paligemma-3b-pt-224`，光有 token 不够 —— 必须用对应账号
  去 https://huggingface.co/google/paligemma-3b-pt-224 接受协议。
  验证：`huggingface_hub.hf_hub_download('google/paligemma-3b-pt-224', 'config.json')`。
- **`--policy.repo_id` 即使不 push 也是必填的。** lerobot 在
  `push_to_hub` 检查之前就 validate 了。塞个占位：
  `--policy.push_to_hub=false --policy.repo_id=local/<run_name>`。
- **相机 key 对齐 pi05_base。** `pi05_base` 声明 3 个相机
  （`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`），
  aswinkumar99 的 SO-101 数据集只有 2 个（`overhead`、`wrist`）。
  CLI 桥接，不改代码：
  ```
  --policy.empty_cameras=1
  --rename_map='{"observation.images.overhead":"observation.images.base_0_rgb",
                 "observation.images.wrist":"observation.images.right_wrist_0_rgb"}'
  ```
- **4B Pi 0.5 在 24GB 显存上的配方。** 四个一起开能舒服塞下：
  `--policy.train_expert_only=true`（冻结 2B VLM，只训 300M action expert
  + projections）、`--policy.freeze_vision_encoder=true`、
  `--policy.dtype=bfloat16`、`--policy.gradient_checkpointing=true`。
  实测：4B 总参 / 693M 可训，4090D 上 batch=1 大约 1.7 steps/s。
- **可以无视的 warning。** 加载时会打
  `Missing key(s) in state_dict: ...language_model.embed_tokens.weight`，
  训练照样正常 —— 是 Pi 0.5 load 路径里的权重重映射残留。

## Simulation Settings 面板（Isaac Sim UI）

视口里那个面板的开关到底干嘛的：

- **USD / Fabric CPU / Fabric GPU**：物理状态后端。Fabric GPU 是默认，
  训练性能必须用。USD 退回每帧 USD 读写，只在排查 USD prim 状态时才用。
- **Reset Simulation on Stop**：点 `Stop` 时是否回 t=0。
  和 RL 训练无关（我们走 env API 重置），编辑场景时保持开着方便。
