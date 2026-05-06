# TII Golden Standards: MCAP to RLDS Pipeline

## 核心架构与时序对齐 (Temporal Alignment)
本转换脚本针对异构传感器采样率（Camera ~30Hz, State ~250Hz, Command ~80Hz）重构了时间同步架构。
强制以 `Head Camera` 的时间戳作为全局主时钟 (Master Clock)，对高频的本体感受数据（Proprioception）与控制指令进行近邻降采样对齐，彻底消除时序断层，完全符合 TII 规则手册对数据时间对齐的硬性要求。

## 验证结论 (Schema Verification)
首帧数据抽样验证已通过。核心张量维度已锁死：
- `observation/image_head_rgb`: shape (720, 1280, 3), uint8 (H264实时解码)
- `observation/joint_positions`: shape (28,), float64
- `action/joint_space_commands`: shape (28,), float64
- 附带完整的 TII 强制状态位 (is_first, is_last, is_terminal) 及 language_instruction ("Move to bedside")。

## 依赖与运行
运行需要 `foxglove-schemas-protobuf`, `av`, `opencv-python`, `tensorflow-datasets` 等核心库。
执行：`python3 mcap_to_rlds.py <input.mcap> <output.tfrecord>`
