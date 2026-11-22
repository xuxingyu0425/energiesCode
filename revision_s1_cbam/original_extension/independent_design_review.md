# S1+CBAM 扩展实验：独立只读设计检查

检查日期：2026-11-17；进度快照为 seed 42 / epoch 69，尚未完成五种子。本检查只读取源码、协议和已有记录，未导入训练模型、未调用 CUDA、未启动推理、未修改 runner/protocol/其他脚本。

**结论：在所查设计和已发生的 seed 42 启动核验范围内，未发现阻断该 S1+CBAM 对照公平性的实现问题。** 设计上的主动变化是：训练回归由历史 S2 对 A/B 原框分别计算后平均，改为原生 CIoU+DFL 对融合框计算一次。现有证据不能扩张为“全部种子/全部训练批次已经实测相同”，也不能推断未来结果。

## 已核实

1. **源文件和路径。** `BASE.parents[2]/experiments` 在当前目录层级解析为 `E:/风机图像/experiments`，没有错位。runner 当前 SHA256 与 `protocol.json` 和 `frozen_sources/runner.py` 一致；协议列出的输入文件 SHA256 均匹配。`wtbd_fused.yaml` 实际指向本工作区 fused 数据，训练记录首批文件路径也正确。

2. **历史 S2 实现被冻结。** runner 在导入历史 attention/core 脚本前，显式把 `reports/yolov8_attention_comparison_v1/frozen_sources/scripts/wtbd_dual.py` 载入 `sys.modules['wtbd_dual']`；冻结文件 SHA256 为 `ec8476eaef4a0f3b558fc461467aab43798514f4619745ad562ff07289b71155`。当前工作版 `experiments/scripts/wtbd_dual.py` 已扩展 `dual_target_soft_class/dual_target_full` 等策略，与冻结版不同；新 runner 没有误用它。冻结版虽也包含可选 distributional/full 分支，但历史 `dual_target` 路径不激活软分类或单边框降权。

3. **初始化。** preflight 保存了 42—46 五个种子的 full/common state 与历史同种子 S2+CBAM 一致性；seed 42 的 `initialization_verified.json` 又在真正训练前核验全 state。架构是相同的 P3/P4 CBAM、3,014,982 参数，含六类检测头和 CBAM 参数，并非只比较 COCO 主干。

4. **增强和样本路径。** 新 trainer 继承冻结 `DualAnnotationTrainer.build_dataset`，训练仍携带 12 个辅助关键点，确保走历史 paired 增强/裁切路径；仅替换 `get_model` 为普通 `DetectionModel`。seed 42 首个增强 batch 的全部 tensor 哈希、图像顺序与历史 S2+CBAM 一致，包含 `img/bboxes/cls/keypoints/batch_idx`。辅助关键点不产生预测分支，也不被原生 detection loss 当作 pose 监督。

5. **分配、分类与 gains。** 直接比对本机 `ultralytics/utils/loss.py` 与冻结 S2：均以融合 `batch['bboxes']` 和相同 resolved class 作 TAL 分配，top-k/alpha/beta 继承同一个原生构造；S2 `dual_target` 不修改分类质量 target，BCE 与质量和分母相同。原生 `max(sum,1)` 与 S2 `clamp_min(1)` 在此均为同一质量归一化。`box=7.5, cls=0.5, dfl=1.5, cls_pw=0.0` 一致；没有新加类别重加权。`cls_pw=0` 的原生 trainer 不施加类别权重。

6. **训练参数。** 对照实际生成的 seed 42 `args.yaml`，与历史 S2+CBAM 的差异仅为 `project` 和 `save_dir` 输出位置。200 epochs、batch16、AdamW、学习率/衰减/warmup、AMP、增强、close_mosaic、workers、patience、deterministic 等相同。runner 不自动减 batch，不 NaN rollback，不从半途权重静默重启。日志中的早期缓存权限启动失败不能当作已运行 epoch 的训练回退；当前正式进度与校验记录已单独存在。

7. **验证与 best。** 验证模式不调用 paired dataset；两边均使用同一个普通 fused 验证集。历史 S2 criterion 在无 keypoints 的验证 batch 上回退原生 loss，新 S1 直接使用原生 loss；checkpoint fitness 由同一原生 validator 产生。故新 best 应称 **原 fused-validation 选择的 best**，不能改称 common-A validation best。epoch 200 是主终点，best 为敏感性比较，协议已写清。

8. **EMA 和权重。** 两者继承相同 trainer 的 `ModelEMA` 创建/更新/属性同步及 checkpoint 保存逻辑；本机源码显示保存的 `best.pt/last.pt` 均取 EMA。普通/dual 模型外层类不同，主要决定 criterion；未见引入额外可训练参数、另一路 EMA 或不同权重选择。不得把 EMA `last.pt` 误描述为未经 EMA 的最后一步 raw 权重。

9. **评价设计。** 已阅读 `evaluate_extension.py` 和 `analyze_extension.py`：需全队列五种子完成、200 行和 checkpoint 哈希核验后才启用 GPU；新的 A validation 阈值先按 0.01—0.99 网格最大 pooled F1 选择，完全相同 F1 时取最小阈值，并在任何该种子 test capture 之前落盘。测试 A/B、best/last 的配置固定；保留实际 800×800 padded 坐标，native imgsz=768/batch16/conf=.001/iou=.7/max_det300 与旧评价一致。B 敏感性统计保持预测不变，只替换 reference。S2 使用历史原始 cache/权重哈希，而非重新选择测试结果。

10. **旧结果 reconciliation。** 已有 `analysis/historical_reconciliation.json` 声明纯 CPU 核对 10 个 S2 checkpoint 的 AP、定位、A/B 筛查和阈值通过；源码确实读取旧 cache、校验权重，并与 v25 源数据比对。该报告是已有核对证据，不代表新 S1 的 test 结果已经存在。

## 必须保留的解释边界

- “相同 assignment/分类”应写作**相同规则、目标来源和超参数**。随着回归梯度使模型轨迹分离，实际预测、正样本分配及分类梯度会自然变化；不能声称整个训练过程中每个 anchor、每次分类 loss 都逐值相同。
- 首批一致是实测范围；后续批次沿用相同 paired 数据管线、seed/worker 配置，但没有本次逐 batch 哈希证明。不应写成已验证所有增强批次完全相同。
- 当前实际启动证据只有 seed 42；其他种子的 preflight 初始化核对与正式训练完成是不同层次。
- 已锁定 YAML、audit、配置、脚本、初始化和首批；没有历史全量图像像素指纹及整个 site-packages 的历史快照。没有发现文件改动的具体证据，不把这一 provenance 范围限制升级成已存在混杂。
- 绝对 `E:/风机图像/...` 源路径可保证本机定位，**不是即取即用的外部复现包**；后续分享时需要清楚的环境与路径设置说明，不宜在进行中的训练中改 runner。
- 本检查不对新 S1 的优劣、显著性或机制效应作预判；新评价需等待完整权重和哈希校验。

证据入口：`protocol.json`、`preflight.json`、`records/seed_42/{initialization_verified,arguments_verified,first_batch,progress}.json`、`runs/seed_42/args.yaml`、历史 `reports/yolov8_attention_comparison_v1` 的冻结源码/同种子记录，以及本机 CV_uav 环境的 `ultralytics/utils/loss.py`、`models/yolo/detect/train.py`、`engine/trainer.py`。
