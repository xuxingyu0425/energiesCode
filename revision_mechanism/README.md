# 标注差异与 S2 作用：返修机制证据包

生成日期：2026-11-17。仅使用已有 CSV、JSON 预测缓存和原始 A 标注 XML，CPU 分析；未启动训练或推理，未更改旧实验、已投稿件或测试阈值。

## 可以据此回答什么

1. 历史同预测梯度诊断显示，S2 与 S1 的监督差异集中于有边界分歧的训练对象，尤其是少量 IoU 较低的配对。
2. 新完成的测试对象分组**没有显示边界分歧越大、定位性能收益就越大**。不能把梯度差异的集中程度解释为 AP/IoU 收益的因果归因。
3. 当前新分组对比是 **S2+CBAM − A+CBAM**，其中正样本分配目标和回归目标均改变。尚待新 S1+CBAM 结果才能在同 CBAM、同平均框分配条件下隔离双边界回归与坐标平均回归的区别。

## 已完成的输入核验

- 对 1,065 个原始 A XML 文件核对 1,584 个框的类别、坐标和唯一映射。
- 发现历史审计表 `object_index` 是“配对对象优先、单标注回退对象后置”的顺序，不总是 A XML 的顺序；共 17 个对象的两个索引不同。已按 **image + A 框精确坐标 + 类别**建立唯一 XML 索引，不能直接按旧 `object_index` 连接预测 `gt_index`。
- 核对 20 个预测缓存（两种配置 × 两个 checkpoint × 五个种子），每个缓存 161 测试图像。预测的 GT 类别/索引与原始 A XML 完全对应。
- 缓存采用 nominal 768 / padded 800 的评价坐标。1024 图像坐标的转换是 `x * 0.75 + 16`，并非直接乘 `800/1024`。最大框坐标误差为 0.00006104 像素（序列化浮点误差）。
- 用原先相同的 confidence≥0.25、同类别、confidence 降序一对一 IoU≥0.5 匹配，重算全部 1,612 行 common-object 结果，与归档 CSV 在绝对误差 1e−12 内一致。
- 将 YOLOv8 的 10 个梯度参考模型、11,320 条对象级梯度记录重汇总至分组/参考模型表，校验总量及各分组差异平方量占比。
- 文件逐一 SHA-256 保存在 `source_manifest.json`；分析检查在 `results/qa.json`。

## 训练标注组成及历史梯度证据

训练集共 1,132 个目标：1,100 个配对、32 个单框回退。配对中 743 个坐标相同（67.55% 的配对；65.64% 的全部目标）；357 个坐标不同（31.54% 的全部目标）。

| 分组 | 训练目标数 | 占全部训练目标 |
|---|---:|---:|
| 坐标相同 | 743 | 65.64% |
| 不同，配对 IoU ≥0.95 | 176 | 15.55% |
| 0.90≤配对 IoU<0.95 | 113 | 9.98% |
| 配对 IoU<0.90 | 68 | 6.01% |
| 单框回退 | 32 | 2.83% |

在相同预测、正样本分配、质量权重和归一化下，配对 IoU<0.90 的 68 个训练目标贡献定位梯度差异平方量的 **59.30%**（S1 参考权重、五种子均值）和 **62.77%**（S2 参考权重、五种子均值）。坐标相同与单框回退分组的差异近数值零。

仅 seed 42 的两个 YOLOv8 参考模型进行了 CIoU/DFL 分量诊断：CIoU 差异项平方量/总差异平方量约 **98.59%–98.84%**，DFL 对应 **0.50%–0.68%**。保留了 signed cross term；这些分量不应直接相加或称作“性能增益百分比”。诊断在 eval 模式、未增强的终点参考模型上进行，不是全 SGD 轨迹重放，也不是双目标 CIoU 对性能的因果分解。

历史文件的 A/B 平均 AP 结果没有用于本次新分析或当前 A 主评估结果表。`historical_records/ANALYSIS_NOTES.md` 保留旧研究全文供审计，含旧 A/B 口径及其他架构，不宜直接作为投稿正文。

## 测试分组：当前 S2+CBAM − A+CBAM

测试集 233 个 A 对象、161 图像、111 个 perceptual-hash 近重复组；其中 162 个 identical、67 个 nonidentical、4 个 unpaired。分组按现有标注定义，不依据模型增益挑选。

主估计是每个 seed 内共同检出对象的均值 IoU 差，再对 5 个 seed 等权汇总。以下 t 区间是固定测试划分下的训练变异，n=5、df=4，未作多重比较调整。

| Checkpoint | 组 | 全测试对象数 | 每 seed 共同对象数范围 | ΔIoU 均值 | 95% seed t CI |
|---|---|---:|---:|---:|---|
| Epoch 200 | identical | 162 | 107–111 | +0.005318 | [0.000132, 0.010503] |
| Epoch 200 | nonidentical | 67 | 46–52 | +0.002005 | [−0.015039, 0.019048] |
| Epoch 200 | unpaired | 4 | 2–3 | +0.052750 | [−0.014485, 0.119985] |
| Original best | identical | 162 | 106–112 | +0.007242 | [0.001200, 0.013285] |
| Original best | nonidentical | 67 | 46–53 | −0.000644 | [−0.025070, 0.023781] |
| Original best | unpaired | 4 | 2–3 | +0.031209 | [−0.006416, 0.068835] |

直接比较 nonidentical 组增量减 identical 组增量：

- Epoch 200：−0.003313，95% seed t CI [−0.016956, 0.010330]。
- Original best：−0.007886，95% seed t CI [−0.027123, 0.011350]。

不能从“某组区间不含零、另一组区间含零”推断两组显著不同；应看上面的直接组差。这里没有证据支持 S2 在有分歧的测试对象上具有更大的收益。

### 单独的测试样本敏感性检查

对 111 个近重复组作 10,000 次 cluster bootstrap，同一抽样权重同步用于五个已训练模型的 seed 对比，保持组内图像/对象在一起，再计算五个 seed 的等权均值。模型权重和共同检出对象集合固定。它描述**已有模型下测试图像组抽样变异**，不是新的训练重复、跨叶片验证、或 joint training-and-population CI。

- Epoch 200 identical 组的 group-bootstrap 95% percentile interval 为 [−0.001422, 0.012164]；nonidentical 为 [−0.010582, 0.012645]。
- Epoch 200 两组直接差 interval 为 [−0.015355, 0.007018]。
- 因此，identical 组 seed CI 不含零并不代表测试样本层面也有稳定正效应。
- 原始测试近重复组少于 10 的小分组不报告 bootstrap 区间；unpaired 仅四对象，保留描述性结果，不据此作机制结论。
- bootstrap 分层结果均为探索性、未作多重比较调整；没有输出或主张确认性 p 值。

## 可用于回复信的英文表述草案

> We agree that a change in the regression objective does not by itself establish a performance benefit. We therefore distinguish the gradient-level effect of annotation discrepancies from their association with localization outcomes. In a fixed-prediction diagnostic on the training set, the 68 paired targets with annotation IoU below 0.90 represented 6.01% of training targets but accounted for 59.30% and 62.77% of the squared localization-gradient differences in the S1- and S2-trained reference models, respectively, averaged over five seeds. This concentration describes changes in supervision, not the proportion of performance improvement attributable to those targets. An exploratory analysis of S2+CBAM versus A+CBAM did not show larger common-object IoU gains for nonidentical test annotations than for identical ones. The direct between-group difference at epoch 200 was −0.0033 (95% seed-paired CI: −0.0170 to 0.0103). We consequently do not claim that the observed localization improvements have been causally explained by annotation disagreement.

这段只能在正式回复中确认已纳入相应图表并给出真实位置后使用；目前没有修改稿件，也没有新增 SI 编号或虚构行号。后续 S1+CBAM 对照若改变解释，应据实际结果重写。

## 文件与复现

- `analyze_annotation_discrepancy.py`：独立 CPU 分析与核验脚本（无 torch、无模型推理）。
- `source_manifest.json`：全部输入路径、原始文件 SHA-256，以及本分析脚本哈希。
- `historical_records/`：原标注记录及经核验筛选的 YOLOv8 梯度源数据；旧 protocol/notes/QA 只作来源审计。
- `results/test_annotation_index_verified.csv`：A XML 与历史配对审计顺序的正确映射。
- `results/cbam_common_objects_with_discrepancy.csv`：核验后的对象级连接。
- `results/discrepancy_localization_per_seed.csv`：每种子对象数、图像数、组数、两模型 IoU 和差值。
- `results/discrepancy_localization_primary.csv`：identical/nonidentical/unpaired 主分组，seed CI 与单独 group-bootstrap CI。
- `results/discrepancy_localization_exploratory_bins.csv`：细分 IoU bins，全部探索性。
- `results/discrepancy_effect_modification.csv`：nonidentical − identical 的直接差。
- `results/historical_cbam_training_times.csv`：已有 S2+CBAM 五 seed 的真实累计训练时间。每次 200 轮约 31.1 分钟；五次累计约 2.59 小时，不含准备、缓存、评价和潜在环境差异，不是新 S1+CBAM 的实测耗时。
- `results/qa.json`：所有核验项目、分析口径及限制。

复现命令（需保留现有工作区数据路径）：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python E:\风机图像\papers\energies_revision_20261117\mechanism\analyze_annotation_discrepancy.py
```

该脚本只在本 `mechanism/` 目录写输出，读取的历史文件保持原样。新 S1+CBAM 结果需经单独缓存/索引验证后接入，不能把本对比重命名为 S2−S1。
