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

## Simulation Settings 面板（Isaac Sim UI）

视口里那个面板的开关到底干嘛的：

- **USD / Fabric CPU / Fabric GPU**：物理状态后端。Fabric GPU 是默认，
  训练性能必须用。USD 退回每帧 USD 读写，只在排查 USD prim 状态时才用。
- **Reset Simulation on Stop**：点 `Stop` 时是否回 t=0。
  和 RL 训练无关（我们走 env API 重置），编辑场景时保持开着方便。
