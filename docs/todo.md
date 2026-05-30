# TODO

## SO-101 env (W2)

- [ ] **Feetech 0% 标定锚点对齐**。当前 `SO101_FEETECH_OFFSET` 用 URDF
  限位的几何中心（wrist_roll=0.0485, gripper=0.785）。真机
  `lerobot-calibrate` 录数据后，如果实际标定把 0% 锚在 URDF 0 rad（机械
  零位），需要把这两个 offset 改成 0，并相应改 scale。录数据前确认。

- [ ] **PD 稳态误差**。stiffness=1000 在 elbow_flex / wrist_flex 上零位
  hold 仍有 20%+ Feetech-norm 稳态误差（重力压塌）。对 Pi 0.5 推理不
  致命（policy 会发非零 hold），但 W3 跑 PPO 时 critic 看 reward 可能
  受影响。届时考虑：
  - 把 stiffness 拉到 2000 看是否收敛
  - 或加 actuator-level gravity compensation
  - 或对 reward 做 vel-based shaping，不依赖绝对姿态

## W2 后续

- [ ] 真机遥操采 sponge pick-and-place 数据集，约 100-300 条 episodes。
- [ ] `lerobot-train` finetune `pi05_base` on the dataset。
- [ ] 写 `LeRobotPi05Wrapper`：env 输出 → `observation.state` /
  `observation.images.*` 键名重命名；image uint8 [B,H,W,3] → float
  [0,1]；action chunk `[B,50,6]` 按时间步索引。

## 未确认事项

- [ ] sponge 物体 USD：目前 env 里 spawn 的是 DexCube，需要换成 sponge
  mesh（或者继续用 cube 当代理物体，等真机数据集确定再换）。
