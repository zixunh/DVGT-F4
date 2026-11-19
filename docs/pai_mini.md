# PAI mini：单个数据块

`tools/data_process/download_pai_mini.py` 从 PhysicalAI 主数据集下载一个数据块，默认要求恰好 100 个有效 clips。下载内容为六相机（不包含 rear tele）、顶部 LiDAR、offline 相机/LiDAR 标定、传感器外参、自车尺寸、自车位姿和障碍物框轨迹。原始 ZIP 保持压缩，后续读取器可按 clip 读取。

这不是 NVIDIA 单独发布的 mini benchmark，也不包含完整的 Foundation4D 稠密深度、图像语义或 occupancy 标签。这里提供原始数据子集的可重现下载；仍需后续数据转换与标注生成，才能供 `stream4d` 训练。

先在 [数据集页面](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) 完成访问授权，再在下载机器、当前用户下登录。网页登录不会自动配置终端凭据。

```bash
.venv/bin/hf auth login

# 只查询公开文件列表和大小，不下载传感器数据。
.venv/bin/python tools/data_process/download_pai_mini.py \
  --output /mnt/raid/dataset/PAI/mini

# 下载及断点续传，使用已保存计划中的同一个数据集 commit。
.venv/bin/python tools/data_process/download_pai_mini.py \
  --output /mnt/raid/dataset/PAI/mini --download --workers 3
```

下载器依赖 `huggingface_hub` 和 `pyarrow`。可使用 `--chunk N` 选择其他数据块；不同块或不同版本应使用不同输出目录，避免混合数据。`--wait-for-auth 900` 可等待当前机器配置登录后自动继续。

在数据集 commit `14ed7a77897edafb113dc86fde8b8173ae7fb028`，chunk 0 的所选内容与元数据共 19 个文件，45,437,548,382 字节，约 42.32 GiB。正式开始传感器下载前，脚本读取索引检查 clip 数量、有效性与所需传感器/标注是否存在。整个数据集约 100 clips 打包一个 chunk；本脚本下载一个 chunk，不会调用全库 `snapshot_download`。

输出包含：

- `download_plan.json`：固定的数据集 revision、文件大小、上游 SHA256、下载状态。
- `clips.jsonl`：100 个 clip ID、原始 `alpamayo_aligned` 划分、Foundation4D 评测集成员标记及 `training_eligible`。
- `verified_files.json`：已完成文件的 SHA256、大小和本地时间戳，用于续传检查。
- 原始 `camera/`、`lidar/`、`calibration/`、`labels/` 以及 `metadata/`、`splits/`。

`training_eligible` 仅在 PAI 原始划分为 train 且不属于 Foundation4D 评测 clips 时为真。一个 chunk 内可能同时含有训练和测试场景；不要把整个 mini 目录当作训练集。微型子集不代表完整数据集的分布或评测成绩。

每个下载文件检查字节数并计算 SHA256；上游提供 LFS SHA256 时进行比对。只有所有文件通过检查，计划状态才写为 `complete`。网络失败后重新执行相同命令即可续传；不要把计划文件存在等同于数据下载完成。
