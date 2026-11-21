# PAI 原始数据到在线 DVGT-F4

入口是 `tools/data_process/prepare_pai.py`，实现位于 `stream4d/pai/`。读取已经下载好的本地 mini，不访问网络、不解压整套数据。`configs/stream4d/pai.yaml` 是六相机、低分辨率的集成配置；默认冻结官方主干，先训练新分支。完整运行记录见根目录 `reproduction_status.json`。

## 安装与转换

```bash
.venv/bin/python -m pip install -e '.[pai]'

# 全部 100 clips，约 2 Hz、96×160；分开保存官方 train/test。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tools/data_process/prepare_pai.py \
  --raw-root /mnt/raid/dataset/PAI/mini \
  --output /mnt/raid/dataset/PAI/mini_stream4d \
  --config configs/stream4d/pai.yaml --workers 4

# 一段 train 和一段 test，每段 8 帧，适合改代码后快速验证。
.venv/bin/python tools/data_process/prepare_pai.py \
  --output /mnt/raid/dataset/PAI/mini_stream4d_smoke \
  --max-clips 2 --max-frames 8 --workers 2

# 独立复核全部 NPZ、官方划分、时间顺序和源/目标帧分离。
.venv/bin/python tools/data_process/check_pai_conversion.py \
  --data /mnt/raid/dataset/PAI/mini_stream4d
```

每个 clip 的 MP4 临时提取到系统临时目录，顺序解码后清理；LiDAR 通过 ZIP_STORED 成员的有界随机访问按一个 row group/单次扫描读取，不把 33 GB 点云包或完整视频放进内存。转换结束生成 `train.jsonl`、`test.jsonl`、`conversion_plan.json` 和 `clips/<clip_id>/frame_*.npz`。单个 clip 成功后写 `conversion.json`；重复同一命令复用完成的 clips。不同分辨率、采样率或网格必须使用新的输出目录。计划只有在所有选定 clips 成功后才标为 `complete`。

采样约为指定 Hz，使用真实相机时间戳，不能假定恰好等间隔。场景之间重置模型状态，不能将这 100 段独立的约 20 秒视频拼成一个连续驾驶场景。

## 数据与监督含义

| 原始数据 | 转换结果 | 训练作用 |
|---|---|---|
| 六路 f-theta RGB、offline 内外参 | 去畸变 pinhole RGB、K、相机 RDF → 车体 FLU | 模型输入；相机顺序固定 |
| 当前版本 Draco 点云的逐点 timestamp | 自车运动补偿后的单次已完成扫描，z-buffer 稀疏 `depth_z` | 几何深度监督；主干冻结时仅记录损失，解冻最后时空块后才更新几何 |
| 同次扫描的命中点和采样射线 | `occupancy_binary`：-1 未知、0 观测自由、1 命中 | 只训练 binary occupancy，不伪造语义类别 |
| `obstacle.offline` | `[cx,cy,cz,dx,dy,dz,yaw,vx,vy]`、12 类、稳定 track ID | 3D 检测与速度监督；标签来自 NVIDIA autolabel |
| 下一相机帧的 RGB、相机位姿和目标框 | `target_rgb`、`target_valid`、世界坐标框 | Gaussian / joint 阶段的留出帧重建目标 |

12 类顺序固定为：`automobile, heavy_truck, bus, train_or_tram_car, trolley_bus, other_vehicle, trailer, person, stroller, rider, animal, protruding_object`。转换器和训练器同时检查映射；不可直接套用默认 `dvgt2.yaml` 的 10 类配置。

框先从标注指定的 `rig@reference_frame_timestamp_us` 转到 clip anchor，再插值到当前参考时间。绝对速度由世界坐标轨迹差分得到，最后旋转到当前自车坐标，不能把自车坐标下的框中心差分当作物体速度。无可靠速度的端点通过 `box_velocity_valid` 屏蔽匹配和回归中的速度项。稳定 ID 由每个 clip 的原始字符串 ID 映射而来；映射保存于该 clip 的报告。

LiDAR 只使用在所有源相机参考时刻之前已结束的最近扫描，并使用逐点自车位姿补偿；没有多帧累计。深度为**相机 z-depth**，不是点到自车原点的距离。二值占据中的未观测区域始终为 -1，射线不穿过测量终点继续标 free，命中优先于 free。占据 bounds/shape 与配置同时固定并检查。

相机时间戳按官方 NCore 转换器解释为读出结束时间，本实现以 `timestamp - shutter_delay_us/2` 近似整帧拍摄时间。参考时刻取六相机中最晚的中点时间，默认最大偏差 30 ms。f-theta 使用 `reference_poly`、`linear_cde` 和官方像素中心约定；反向多项式为参考时数值求逆，而非把近似正向多项式当成精确投影。自车遮挡 mask 与去畸变有效范围共同屏蔽深度和重建监督。

`world_from_ego` 以当前转换片段第一帧自车坐标为世界原点；源与目标框、相机位姿使用相同世界坐标。留出目标通常是源帧之后约 1/30 秒的帧，其 RGB 不进入主干。目标时间不能当作源输入时间；在线推理只消费当前源 RGB 与标定，不消费 GT 深度、框或未来 RGB。

## 验证与训练

```bash
# 原始数据 -> 三个训练阶段 -> held-out test 推理/渲染 -> 主干梯度与缓存检查。
bash tools/stream4d/pai_smoke.sh

# 独立验证真实数据上的最后一个时空模块微调。
.venv/bin/python tools/stream4d/validate_pai.py \
  --train-manifest /mnt/raid/dataset/PAI/mini_stream4d_smoke/train.jsonl \
  --test-manifest /mnt/raid/dataset/PAI/mini_stream4d_smoke/test.jsonl \
  --finetune-blocks 1 --steps 5 --output outputs/pai_validation

# 完成全量转换后进行新分支训练。steps 是实验预算，不是论文配方。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash run_f4.sh train \
  --config configs/stream4d/pai.yaml \
  --manifest /mnt/raid/dataset/PAI/mini_stream4d/train.jsonl \
  --stage perception --steps 1000 --output outputs/pai_perception
```

验证脚本断言：官方权重严格加载；检测与二值占据参数发生更新；解冻时主干获得梯度且更新；没有语义标签的图像/占据/BEV 分类器既无梯度也不被 AdamW decay 改写；测试 clip 与训练 clip 分离；跨窗口后缓存大小稳定。CLI 禁止将 PAI `test` 或 Foundation4D 评测 clips 用作训练。checkpoint 的 `trained_tasks` 记录真正得到可微监督的任务，不能只凭 `trained_stages` 推断全部分支都已训练。

96×160 六相机的短训练只验证接口和优化链路。训练日志降低不等于泛化成绩；现有深度 MAE 是该稀疏投影协议下的诊断值，不是 Foundation4D dense-depth benchmark。更高分辨率可通过 `--size 192 320` 重新转换，显存与吞吐需要重新测量。

当前机器已经完成 100 clips 转换及逐 NPZ 检查：3,250 train 帧、574 test 帧，合计 3,824 帧，NPZ 约 2.17 GB。全部 85 个 train clips 各取前两帧进行了 170 步感知 warmup，再各取第一帧进行了 85 步 Gaussian warmup；清单和配方在 `outputs/pai_mini_warmup/`，权重为 `gaussian/adapter.pt`。全部 15 个 test clips 的 574 帧在线推理完成，场景切换正确重置状态，缓存上限为 260,997,120 字节。默认阈值下有 350 帧产生跟踪结果，最大 52 条轨迹；这里只验证运行与状态预算，没有评估跟踪精度。独立的最后一个时空模块微调检查峰值显存约 8.02 GiB。完整数值及验证限制保存在 `reproduction_status.json`，训练输出和数据不提交到 Git。

## 仍未覆盖

- 图像语义、语义占据、BEV 语义没有标签，这三个分类器仍未训练。需要独立生成训练 split 的伪标签或接入合法的训练标签。
- 单次 LiDAR 投影不是 Foundation4D 多帧动态聚合和稠密化 autolabel；没有物体运动 deskew、稠密补全或法线标签。
- 没有滚动快门逐行补偿或跨相机时间差的图像重投影；采用有界偏差和中点近似。当前 mini 的 external distortion 均为空，遇到未知畸变会显式拒绝。
- 低分辨率、粗网格和短 warmup 不代表正式 mAP、mIoU、HOTA 或重建成绩；正式 evaluator 和收敛训练仍需后续实验。

格式依据：[PAI 开发套件](https://github.com/NVlabs/physical_ai_av)、[NCore PAI 转换器](https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py)、[NCore 传感器模型](https://nvidia.github.io/ncore/data/sensor_models.html)。研究目标参考 [Foundation4D](https://arxiv.org/abs/2610.30340)；本实现没有声称复制其全部标注管线。
