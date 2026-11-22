# 复现材料阶段验收：2026-11-17

本目录是原投稿结果的可核查代码与数据包，外加新 S1+CBAM 扩展的代码接口和协议快照。
本阶段没有改论文，没有启动训练，没有使用 GPU，没有把未完成的新实验计入原结果。

## 已实际通过

1. **历史材料覆盖**：40 份原训练参数、40 份完整 200-epoch 日志；80 个原 best/last
   checkpoint 均在原工作区找到并计算 SHA256。权重仅列清单，未复制大文件。
2. **来源和依赖**：306 条复制来源指纹核对通过；补齐本地 Python 依赖和第三方注意力
   实现/许可证；S2 使用 `ec8476...` 的历史冻结版本，而非后来加入 soft-class 分支的工作副本。
3. **精确划分**：1065 图像，744/160/161；750 个 pHash 组不存在跨集合分配。
4. **筛查重放**：35 runs × validation/test = 70 组，重新置信度排序、同类一对一 IoU≥0.5
   匹配，P/R/F1、TP/FP/FN 与归档结果最大绝对误差 `1.4210854715202004e-14`。
   核对已存 99 点验证网格、最大 F1 及低阈值破同分规则；逐 run 的最优点重匹配得到验证。
   另外对 A/42 从预测缓存重新计算整个 99 点曲线并选出 0.51。
5. **表格**：可导出原 Tables 1–14 的数值/定义。原 Table 2–14 共 101 行、841 个显示数字，
   与当前原投稿 `manuscript.tex` 按显示精度逐项一致；Table 1 为监督定义表。
6. **绘图**：已运行原 Figure 2–5 的数值与示例绘制入口。Figure 4 四个案例的 image/gt index/
   两模型 IoU/差值及候选集合计数与原记录全部一致。没有主张字体、渲染器和导出像素逐字节一致。
7. **模型 CPU 检查**：在 CV_uav 的 Python 3.12.13、torch 2.3.1+cu121、Ultralytics 8.4.37
   环境中，baseline/ESB/CBAM CPU 构建和 64×64 前向有限，参数量分别为
   3012018/3043768/3014982；S1 配对管线模型的 criterion 确认为原生 v8DetectionLoss。
8. **初始化 CPU 重建**：用原 COCO 预训练文件（f59b3d...）重建五个 CBAM 初始模型，
   五 seed 的完整 state、共同 state 和参数量均与历史 S2+CBAM 初始化记录的 SHA256 一致。
9. **扩展代码保留**：原 runner/protocol/preflight 原样拷入 provenance/revision_s1_cbam；
   五 seed 历史初始化及首批张量记录保留在 provenance/original_attention。

## 仅静态或 dry-run / 尚未声称完成

- `tools/train_one.py` 的原配置及新 S1_CBAM 入口只做 plan/dry-run；全训练由目录外经审计的
  新扩展 runner 进行，本包未另起训练或 GPU 任务。
- S1_CBAM portable 入口具备完整初始 state 与首批张量核验 hook，但其实际训练 hook 未由本阶段
  重新运行。当前活动扩展 runner 的验证状态以其自身 records 为准。
- `prepare_data.py` 可调用原准备算法并强制匹配原划分，但本阶段未再次建立全部图像/标签目录。
- `evaluate_checkpoint.py` 是显式路径、版本保护的再推理入口；本阶段未进行 checkpoint 再推理。
  所验证的是归档缓存与统计计算，不是新推理或重训精度复现。
- CV_uav 的附加包清单是 11 月 17 日仍保留环境的实测 inventory；只有原 manifest 中的核心版本
  能独立追溯到原实验，不能称所有可选依赖从历史起一直未变。
- Figure 1 是人工绘制的架构示意图，修改后的可编辑设计源待新图完成后纳入。
- 新 S1+CBAM 五种子结果尚未齐备，不预填实验成绩、显著性或结论。
- 公共仓库地址/DOI、作者代码整体发布许可证、原 checkpoint 下载渠道尚未虚构或发布。

## 主要验收证据

- `outputs/package_audit.json`
- `outputs/replay/verification.json`
- `outputs/tables/verification.json`
- `outputs/manuscript_number_audit.json`
- `outputs/figure_selection_audit.json`
- `outputs/model_cpu_check.json`
- `outputs/initialization_check.json`
- `provenance/weight_manifest.csv`
- `provenance/missing_materials.json`

原稿未改动。可在新 S1+CBAM 五种子完成后，以独立版本接入配置、权重清单、预测、统计和图表，
保留原结果与新增返修结果的来源差别。

硬件：作者已确认原实验使用同一台电脑，已纳入 root 保存的只读硬件记录：i9-13980HX、
24C/32T、约32GB内存、RTX4060 Laptop约8GB显存、Windows11家庭中文版64位。当前OS build不冒充历史build。
