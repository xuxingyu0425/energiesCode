# WTBD 机制诊断：结果、最小对照与审计记录

## 诊断结论（2026 年 9 月 7 日）

三项诊断全部完成，没有启动新训练。原实验权重、训练脚本及主结果保持不变。

1. 训练集 1,132 个目标中，有 1,100 对标注；743 对坐标完全一致，占配对目标 67.55%。357 个目标真正存在双框坐标差异，占全部训练目标 31.54%；另有 32 个单框回退目标。小目标、细长目标的框分歧比例更高，但不能据此判断标注错误。
2. IoU < 0.9 的 68 个目标仅占训练目标 6.01%，却贡献四类架构/参考组合中 39.54%–62.77% 的定位梯度差异平方量（各组合取五种子均值）。因此 S2 的新增定位信号集中，而不是对所有目标广泛改变监督。
3. 全训练集、四个 seed-42 参考模型的损失拆分显示：CIoU 差异项平方量 / 总差异平方量为 98.59%–98.88%；DFL 项为 0.50%–1.10%；另有 0.16%–0.74% 的交叉项。这里说的是梯度差异，不是性能收益比例，且不能排除训练早期 DFL 的累积作用。
4. 全部 100 个检查点、400 个 A/B 验证/测试端点评估完成。S2 相对 S1 的测试收益没有跨阶段稳定保留。

| 架构 | 历史 best 的 S2−S1 | 固定第 200 轮的 S2−S1 | 统一验证集四候选选择 |
|---|---:|---:|---:|
| YOLOv8n | +1.134 pp，95% CI [−1.544, +3.812] | −0.036 pp，95% CI [−2.644, +2.573] | 全部选第 200 轮，与 last 相同 |
| YOLO11n | +0.566 pp，95% CI [−1.261, +2.393] | −0.295 pp，95% CI [−2.944, +2.355] | 全部选第 200 轮，与 last 相同 |

固定第 51/101/151/200 轮的测试配对均值依次为：YOLOv8n −1.133/−1.086/+0.621/−0.036 pp；YOLO11n −0.215/+0.542/+0.434/−0.295 pp。八个固定阶段比较的描述性五种子区间均跨零。统一规则与 last 完全重合，不能当作两份独立证据。

固定末轮，YOLO11n 在验证集上五个种子均改善（均值 +1.294 pp），但测试集只有 2/5 改善（均值 −0.295 pp）。这表明除了选点敏感性，还存在验证—测试收益不一致。其原因未被本轮诊断识别；不能宣布已证明“选点偏差”或“真实噪声”。

当前论文可支持的表述：**S2 改变了少量标注分歧样本的定位监督，主要差异来自双目标 CIoU；但在当前数据划分与两种架构上，这种变化没有稳定转化为优于硬平均的测试收益。** 当前 S2 也不处理类别冲突，不能扩大为通用标注去噪方法。ESB 的模块功能重叠假设仍未证实。

## 建议的最小机制训练对照（尚未启动）

优先只为 YOLOv8n 新增 **S2-CIoU-only × 5 种子（42–46）**，与已有 S1 和完整 S2 比较。新臂使用双目标 CIoU、硬平均目标 DFL 和原分类损失；相对于完整 S2，仅改变 DFL 目标形式。初始化、双框数据加载、增强、正样本分配、归一化、损失增益和 200 轮日程均须对齐。

这五次训练回答“主要作用是否由双目标 CIoU 保留，双目标 DFL 是否提供额外作用”。预先固定统一的验证选点规则，并列第 200 轮，报告五种子配对差及区间。若区间宽，结论是证据不足，不能把无显著差异当成等效。

选择 YOLOv8n 是探索性优先级：本轮其定位梯度差异更大，原 best 下 S2−S1 的点估计也较大；不代表机制已经跨架构成立。若需要跨架构机制结论，再给 YOLO11n 增加相同 5 次，合计 10 次。暂不扩展 ESB 或人为加噪的大组合。

人工复核分歧可先于训练进行，但应盲于模型收益，区分合理边界多解与实际标注错误。若论文不再提出额外机制或稳定优越性主张，也可以保留当前完整诊断，不强行加训练。

配套证据：`annotation_distribution.csv`、`annotation_geometry.csv`、`gradient_reference_summary.csv`、`gradient_bin_summary.csv`、`gradient_component_summary.csv`、`checkpoint_effect_summary.csv`、`selection_comparison.csv`、`analysis_qa.json`。完整图表叙述保存在 `artifact.json`，原始及规范化检查点表均保留。

以下为分析口径、可复现性和呈现审计说明。

## Scope and decision

Complete annotation distribution, same-prediction S1/S2 loss and gradient diagnostics, and saved-checkpoint sensitivity before deciding whether another mechanism-training arm is justified. No optimizer steps, training jobs, checkpoint replacement, or revised primary endpoint are authorized or performed.

The existing study and its primary contrasts remain frozen. The new diagnostics are exploratory and cannot retroactively make the original five-seed comparisons confirmatory. This report is for a technical research audience; it is not a submission-ready manuscript section.

## Analysis definitions

- Annotation unit: one row of the existing matched-object audit. The denominator for paired agreement excludes unmatched annotations. The denominator for contribution to training includes all 1,132 training objects. Missing B labels are not counted as perfect agreement.
- Exact agreement means numerically identical box coordinates; it does not mean independent annotators, absence of labeling mistakes, or agreement with adjudicated truth.
- IoU bins: identical; non-identical with IoU >= 0.95; 0.90 <= IoU < 0.95; IoU < 0.90; unmatched. Bin boundaries and geometry thresholds were fixed before gradient results.
- Geometry: aspect = max(w/h,h/w), elongated >= 5; small = area/image area < 1%. Geometry is repeated using A and B boxes; missing B is excluded from B geometry.
- Primary gradient diagnostic: native gained localization loss derivative with respect to pre-DFL box logits, not model parameters, using all 744 training images at the 20 last EMA reference checkpoints. Same predictions, assignment, normalizer, and classification target are held fixed. A reference trained with S1 and a reference trained with S2 both compare the same two objectives; reference training method is not the objective being compared.
- Relative difference: sqrt(sum ||g2-g1||^2 / sum ||g1||^2), concatenating separate batches' gradient coordinates. It is not a full-dataset SGD gradient, and neither its magnitude nor its cosine is a performance effect size.
- Difference-energy share: sum ||g2-g1||^2 in an object group / corresponding sum over all training objects, calculated per reference then summarized equally over five seeds. It is not a percentage of AP improvement or a causal attribution.
- Secondary parameter gradient: all model parameters and full objective, frozen BN, 32 fixed train images balanced across four image-level disagreement strata, seed 42 only, 16 batches per reference. These images are not a prevalence-representative sample; these numbers cannot be directly subtracted from the full-training head-gradient values.
- Supplementary component probe: all training images, seed-42 S1/S2 references in both architectures, separating native gained box and DFL gradient differences. The signed cross term is retained because component energies are not additive by themselves. This probe was specified after observing the primary gradient result, before its own results, and is explicitly exploratory.
- Checkpoints: epoch50/100/150 filenames represent completed epochs 51/101/151; last is completed epoch 200. Historical best remains separate. Val/test A and B are evaluated identically; A/B mean means arithmetic mean of two AP values, not pooled detections or two independent test datasets.
- Common validation selection: retrospectively choose among the four fixed epochs using mean A/B validation AP50:95; resolve ties by earliest epoch, then read the corresponding test score. Never select using test scores. This secondary four-candidate selector is not an equal-size recreation of historical best selection over all epochs. Both historical S1 and S2 selected on fused validation labels; do not claim their selection labels differed.
- Five paired seeds, descriptive t confidence intervals, no new confirmatory significance claims. A zero-crossing interval is not proof of equivalence. Training seeds do not quantify uncertainty from choosing another wind farm or data split.

## QA gates

Input and frozen diagnostic-script SHA256 verification; 1,584 unique annotation objects; 20 complete references, each 93 batches/744 images/1,132 objects; exact object joins; finite gradients; sum of per-object gradient energies reconciles to batch energy; equal assignment and classification losses; equal-target loss control; parameter sample identity; component-energy reconciliation with signed cross term; 100 checkpoint identities/400 endpoints; 80 reused endpoint metrics match the frozen formal study exactly; validation-only selector; full paired-seed coverage.

Windows label-cache creation required a sandbox-external continuation because Python's internal ThreadPool creates a Windows pipe even with dataloader workers=0. This resumed the same frozen checkpoint evaluator and did not change training weights. Original log is retained; continuation log is separate.

Final metadata QA correction: installed Ultralytics `engine/trainer.py:845` copies LAST's full `train_results` into BEST during `final_eval`. Thus the frozen evaluator's raw `completed_epochs=200` on BEST describes the end of the copied history, not the epoch of the BEST weights. Raw files are preserved for audit. `checkpoint_raw_normalized.csv` separates `history_end_epoch` and sets BEST's actual completed epoch to unknown. Fixed periodic/last epochs remain verified. `best_epoch_summary.csv` explicitly reports zero known BEST epochs; no epoch distribution is claimed for BEST. This correction has no effect on metrics, fixed-checkpoint comparisons, or common-validation selection.

## Report chart map and delivery

Selected delivery: one native MCP report, with machine-readable local supporting evidence. No parallel HTML report unless native rendering fails. Technical summary is first after the visible title; scope is defined before detailed findings; methodology, limitations, next steps, and unresolved questions remain visible. A compact audit preview shows the first 10 reviewed gradient-reference rows.

1. Annotation distribution: zero-baseline bars, training object count by five mutually exclusive bins. Retain denominator and fraction in the dataset. Supports concentration/prevalence, not annotation correctness.
2. Gradient composition: grouped zero-baseline bars by disagreement bin and architecture, mean over both reference types and five seeds; retain reference-level values in the underlying source file. Supports concentration of gradient differences; source-references and full-parameter diagnostics are shown separately in a compact table.
3. Checkpoint sensitivity: grouped signed bars for paired S2-S1 test AP differences at 51/101/151/200, with explicit zero baseline. Only four observed checkpoints exist, so a smoothed/dense training-test trend is intentionally omitted. Five-seed confidence intervals and win counts are in the adjacent evidence table; bars alone do not establish significance.

Use neutral descriptive chart titles, clear counts vs percent vs AP percentage-point units, and visible architecture legends. Each chart/table has an adjacent explanatory paragraph and canonical local-file source metadata. Preserve source paths in metadata rather than cluttering reader prose. No synthetic samples or extrapolated test trajectories.

The native artifact validator rejected the component probe's Python producer as a table SQL source, while accepting the other Python analysis sources. The redundant four-row native component table was omitted rather than inventing SQL provenance. Its complete numerical range, denominator, signed cross term, and limitation remain in the source-linked component narrative; all 12 exact component rows and original Python producer remain in the local CSV/source metadata. Three native charts and five native tables remain. The report payload passes the native validator before its single visible rendering call.
