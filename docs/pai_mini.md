# PAI mini：单个数据块

`tools/data_process/download_pai_mini.py` 从 PhysicalAI 主数据集下载一个数据块，默认要求恰好 100 个有效 clips。下载内容为六相机（不包含 rear tele）、顶部 LiDAR、offline 相机/LiDAR 标定、传感器外参、自车尺寸、自车位姿和障碍物框轨迹。原始 ZIP 保持压缩，后续读取器可按 clip 读取。

这不是 NVIDIA 单独发布的 mini benchmark，也不包含完整的 Foundation4D 稠密深度、图像语义或 occupancy 标签。这里提供原始数据子集的可重现下载；仍需后续数据转换与标注生成，才能供 `stream4d` 训练。

先在 [数据集页面](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) 完成访问授权，再在下载机器、当前用户下登录。网页登录不会自动配置终端凭据。

```bash
.venv/bin/hf auth login

# 只查询公开文件列表和大小，不下载传感器数据。
.venv/bin/python tools/data_process/download_pai_mini.py \
  --output /mnt/raid/dataset/PAI/mini --chunk 3

# 下载及断点续传，使用已保存计划中的同一个数据集 commit。
HF_HUB_DISABLE_XET=1 .venv/bin/python tools/data_process/download_pai_mini.py \
  --output /mnt/raid/dataset/PAI/mini --download --workers 8 --range-workers 8
```

下载器依赖 `huggingface_hub` 和 `pyarrow`；启用 HTTP 分段传输时还需要 `requests`。可使用 `--chunk N` 选择其他数据块；续传不传 `--chunk` 时使用已保存的块编号。不同块或不同版本应使用不同输出目录，避免混合数据。`--wait-for-auth 900` 可等待当前机器配置登录后自动继续。

在数据集 commit `14ed7a77897edafb113dc86fde8b8173ae7fb028`，本次选择 chunk 3：100 个 clips 的所需数据齐全，官方 `alpamayo_aligned` 划分为 85 train / 15 test，均不在 Foundation4D 评测子集中。所选内容与元数据共 19 个文件，45,875,590,722 字节，约 42.72 GiB。最初检查的 chunk 0 有 4 个 clips 缺少所需数据，因此未下载其传感器文件。正式开始传感器下载前，脚本读取索引检查 clip 数量、有效性与所需传感器/标注是否存在。整个数据集约 100 clips 打包一个 chunk；本脚本下载一个 chunk，不会调用全库 `snapshot_download`。

输出包含：

- `download_plan.json`：固定的数据集 revision、文件大小、上游 SHA256、下载状态。
- `clips.jsonl`：100 个 clip ID、原始 `alpamayo_aligned` 划分、Foundation4D 评测集成员标记及 `training_eligible`。
- `verified_files.json`：已完成文件的 SHA256、大小和本地时间戳，用于续传检查。
- 原始 `camera/`、`lidar/`、`calibration/`、`labels/` 以及 `metadata/`、`splits/`。

`training_eligible` 仅在 PAI 原始划分为 train 且不属于 Foundation4D 评测 clips 时为真。一个 chunk 内可能同时含有训练和测试场景；不要把整个 mini 目录当作训练集。微型子集不代表完整数据集的分布或评测成绩。

受限仓库在登录前可能以星号隐藏 LFS 哈希，脚本会在授权通过后重新获取固定版本的真实校验值。每个下载文件检查字节数并计算 SHA256；上游提供 LFS SHA256 时进行比对。只有所有文件通过检查，计划状态才写为 `complete`。网络失败后重新执行相同命令即可续传；不要把计划文件存在等同于数据下载完成。

对于本机上无进度的 Xet 大文件传输，示例使用 `HF_HUB_DISABLE_XET=1` 走 Hub HTTP 续传。`--range-workers 8` 对至少 256 MiB 的文件使用 256 MiB 固定分段；至少 8 GiB 的大文件采用指定并发，视频每个文件最多 4 路。已有 Hub 临时文件按可信文件哈希匹配，导入其最长非稀疏下载前缀，避免 Hub 生成新临时文件时重复下载。各段可恢复，并在合并时核验完整 SHA256。服务端必须返回正确的 206/Content-Range，否则拒绝写入；磁盘预算包含所有临时分段及合并所需空间。更小的文件仍走 Hub 下载。

本次远程下载在 `zixun-compute` 的独立 tmux 会话 `pai-mini` 运行，输出目录中的 `run_download.sh` 负责启动任务并写入 `download.exitcode`。断开本地 SSH 不影响该会话；远程服务器需要保持开机联网。完成以 `download_plan.json` 的 `status: complete` 和退出码 0 为准。查看日志：`tail -f /mnt/raid/dataset/PAI/mini/download.log`。
