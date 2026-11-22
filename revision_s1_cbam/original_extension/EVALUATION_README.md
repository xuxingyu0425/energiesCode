# S1+CBAM 返修扩展：评价与统计接力

本目录的训练由 root 的 `runner.py` 管理；下列评价工具不修改 runner、训练协议、runs 或旧实验记录。评价方案在读取新 S1 测试结果之前固定于 `evaluation_protocol.json`。

## 启动命令

GPU 正在训练时可启动等待进程：

```powershell
& 'E:\server\anaconda3\envs\CV_uav\python.exe' -u 'E:\风机图像\papers\energies_revision_20261117\s1_cbam\evaluate_extension.py' wait-and-run --poll-seconds 15 --max-wait-hours 24
```

`wait-and-run` 在训练阶段只读取状态、查询 queue PID、记录等待时间，不导入 torch/Ultralytics、不调用 CUDA。只有 `queue_status.json` 为 `training_complete` 且五个 seed 的完成标记、200 行训练记录、best/last 文件哈希均正确后才开始评价。训练 failed、已退出的 queue PID、等待超过 24 小时或任何评价校验失败都会写 `evaluation_status.json: failed` 并明确退出，保留所有已有缓存；不调整 batch、模型、超参数，也不重启训练。

Windows 等待期间仅设置本线程临时 `ES_CONTINUOUS | ES_SYSTEM_REQUIRED`，不保持显示器开启，不改变系统电源计划，退出时恢复 `ES_CONTINUOUS`。

如果训练已完成，可直接使用：

```powershell
& 'E:\server\anaconda3\envs\CV_uav\python.exe' -u 'E:\风机图像\papers\energies_revision_20261117\s1_cbam\evaluate_extension.py' evaluate
```

只复核历史数据（纯 CPU，无新推理）：

```powershell
& 'E:\server\anaconda3\envs\CV_uav\python.exe' 'E:\风机图像\papers\energies_revision_20261117\s1_cbam\analyze_extension.py' verify-historical
```

## 固定评价口径

- 主比较：**S2+CBAM − S1+CBAM**，五个相同 seed（42–46），固定 epoch 200、A 测试参考；主指标 AP50–95 与在两个模型共同检出 A 对象上的 mean IoU 差。
- 次要：original best（训练时由 fused 验证标签选择）、B 参考、AP50/AP75、TP IoU、IoU 覆盖率、A-val 锁定阈值的 pooled P/R/F1 与 FP/FN。
- 原始 A/B AP 分开报告，不以 A/B 均值充当 A 主结果；不按测试结果选择 checkpoint 或阈值。
- Native evaluator：`imgsz=768, batch=16, workers=0, conf=.001, iou=.7, max_det=300, half=False`。同原评价的 rectangular padding，必须观测到 800×800 评价坐标、A test 161 图/233 GT、B test 161 图/229 GT。
- 定位与共检对象：confidence≥0.25，confidence 降序，同类、一对一、IoU≥0.5。coverage 为每个 GT 的最大同类预测 IoU≥.5/.75/.85 比例，不要求一对一；不要混为 TP recall。
- 每个 S1 seed 首先 capture epoch200 A-validation（160 图/219 GT），在 .01–.99 共 99 个阈值上最大化 pooled F1；完全相同 F1 时取最小阈值。将阈值及 weight/cache/protocol SHA 写入 selections 后，才 capture 该 seed 的测试集。
- B 筛查敏感性保持 A 预测与 A-val 阈值不变，只替换参考框/类别，不在 B 上重选阈值。
- 差值先按 seed 配对，用未四舍五入数据计算，two-sided Student t 95% CI，n=5、df=4。它表示固定划分的训练变异，不是跨叶片/风场总体区间。次要与分组结果为探索性、未作多重比较调整；不按哪些区间不含零挑选报告。

## 历史口径校准

全部训练结束后、第一次新 S1 GPU 评价之前，先在同一 GPU 环境重评历史 `S2_CBAM seed42 last A-test`：

- 对照 v25 A 结果表的 native P、R、AP50、AP75、AP50–95，fraction 单位最大绝对误差须小于 2e−5；
- 同时确认 161 图、233 GT、800×800；
- 历史权重字节 SHA 必须等于原缓存记录；
- 保存 `evaluation/calibration_S2_CBAM_42_last_A_test.json`，失败则停止，不放宽误差门槛或改变模型。

之后共 25 次新 S1 capture：每 seed 的 A last val，加 A/B × best/last 四种 test。连同一次校准共 26 次 native capture。

## 已完成的纯 CPU 预检

- 在 CV_uav 中连续两次 `prepare-only` 均通过；历史数据的真实 native AP、定位、A/B pooled 筛查、99 点阈值选择及 A+CBAM/S2+CBAM common-IoU 配对区间重算一致。
- 历史 S2 五种子的 best/last 权重哈希与相应缓存核对。
- Windows PID 存活查询（当前训练 queue、当前进程、一个不存在 PID）及临时 execution state 设置/恢复均测试通过。
- 等待模块导入后无 torch/Ultralytics 导入，也没有 CUDA 调用。
- 科学评价终点相对最初 CPU-only 方案保持一致。两份早期协议草稿存于 `evaluation/protocol_drafts/`，只记录 JSON tuple/list 比较修复与 watcher 基础设施完善；没有在观察新测试结果后修改终点。
- 具体检查记录：`evaluation/preparation_qa.json`、`analysis/historical_reconciliation.json`。

## 完成标记和主要输出

成功日志最后应包含：

```text
S1_CBAM_EXTENSION_ANALYSIS_COMPLETE
S1_CBAM_EXTENSION_EVALUATION_COMPLETE
```

并产生：

- `extension_complete.json`、`evaluation_status.json` 中 `status=complete`；
- `evaluation/cache/`：25 份 unrounded native 预测缓存，具 weight、labels、evaluation protocol SHA；
- `evaluation/selections/`：五个先于测试固定的 A-val 阈值及完整曲线；
- `evaluation/native_test_raw.csv`：20 个新 S1 的 A/B best/last native 结果。这里 native P/R 是 native evaluator 的均类指标，不能与下面 pooled locked P/R 混用；
- `analysis/native_AP_raw.csv` 和 `native_AP_summary.csv`：S1/S2 的分参考、checkpoint、seed AP；
- `analysis/localization_per_seed.csv`：TP IoU、覆盖率与 TP 数；
- `analysis/locked_screening_per_seed.csv`、`per_class.csv`、`per_image.csv`：run-specific A-val 阈值锁定的 pooled 操作点结果；
- `analysis/S2_minus_S1_common_objects.csv`：两模型重新计算的共同对象交集、左右 IoU 和差值；
- **`analysis/paired_S2_minus_S1_contrasts.csv`**：所有预定配对差、95% CI、5 seed 同向次数和均值；
- `analysis/analysis_provenance.json` 和 `schema.json`：口径、来源及限制；
- 如 mechanism 中正确 XML 索引映射可用，另输出 `S2_minus_S1_common_objects_discrepancy.csv` 和逐 seed/分组汇总。分组区间仍以 seed 为 n；不把对象行当独立 seed，也不预先声称 S2 对分歧对象更好。

若只有 4/5 或更少 seed 在极小某分组共同检出对象，则该分组不强行生成 df=4 CI；保留其可用 seed 数与缺项，整体主比较仍要求完整五 seed。

脚本均只在本新 `s1_cbam/` 下写文件。旧 producer 源码为审计复制到 `evaluation/frozen_dependencies/`，未直接运行旧脚本 main，不会覆盖旧 source_data。
