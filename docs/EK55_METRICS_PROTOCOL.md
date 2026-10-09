# EK55 指标协议与本项目实验边界

## 1. 两组指标不是同一个任务

### 短期动作预测：Action Top-1 / Top-5

- 任务：在目标动作开始前 1 秒预测即将发生的一个动作。
- 类别：EK55 训练标注中出现的 2513 个 `(verb, noun)` 联合动作类。
- 划分：采用 RULSTM 发布的 232 个训练视频和 40 个验证视频。
- 输入：14 个时间点，间隔 0.25 秒；最后一个观测点不晚于动作开始前 1 秒。
- 指标：验证动作的真实联合类别是否位于模型排序的前 1/5 名，按样本计算 micro accuracy。
- 对应脚本：
  - `evaluation/prepare_ek55_anticipation.py`
  - `feature_extraction/extract_ek55_anticipation_rgb.py`
  - `HandObject/train_anticipation_topk.py`
  - `scripts/run_ek55_anticipation_topk.sh`

如果现有 JPEG 是 participant-level 扁平目录，优先运行
`scripts/run_ek55_topk_rulstm_features.sh`：它下载 RULSTM 官方 6.1 GiB TSN-RGB LMDB，
转换出 3.5s→1.0s 的 11 点无泄漏序列后训练同一个 2513 类 temporal classifier。该结果
遵循相同 Top-k 标签/划分/τa 协议，但必须把特征源标为 `RULSTM TSN-RGB`，不能写成
`EgoVideo`。

使用官方完整标注的清单验收结果：2513 action classes；23493/4979 条 raw train/val
actions；按 4.25s→1.0s 的完整历史要求过滤后为 23431/4970 条。

旧的 `HandObject/evaluate_topk.py` 是“动作片段内部特征上的当前动作识别”，并且只覆盖
1263/2490 个随机验证样本。其 3.56%/11.16% 不能填入论文的短期动作预测表。

### 长期未来动作集合：P=25/50/75 All/Freq/Rare mAP

- 任务：观察完整视频的前 25%、50% 或 75%，预测剩余视频中会出现哪些 verb。
- 标签：剩余部分所有 future verb 的 125 维 multi-hot 向量。
- 过滤：每个切分点必须至少有 3 个过去动作和 3 个未来动作。
- 划分和时间点：采用 EGO-TOPO 的 EK55 S1 文件；训练 P=20/30/40/50/60/70/80，
  验证 P=25/50/75。
- Freq：官方 `EPIC_many_shot_verbs.csv` 中训练出现次数大于 100 的 verb；Rare 为其余
  在验证标签掩码内的 verb。
- mAP：先对每个 verb 跨验证样本计算 AP，再在 All/Freq/Rare 类别内取均值。每个 P
  使用整套验证集生成的固定类别掩码；某个 P 没有正样本的已评测类别贡献 AP=0。
- 对应脚本：`evaluation/train_ek55_future_verb_map.py`、
  `scripts/run_ek55_future_verb_map.sh`。

官方 split 验收结果：train 205 videos / 20730 actions，val 67 videos / 7742 actions；
全部 28472 条均能映射回 EK55 官方 UID。协议过滤后得到 1251 个训练阶段样本和 172 个
验证阶段样本（P25=57、P50=60、P75=55），最终可用数还会受本地特征覆盖率限制。

当前 mAP 脚本复用已有 Stage-1 RGB action feature，只是一个“官方标签/划分/指标协议下的
feature-limited baseline”。它不等价于论文中的 EgoVideo 全视频特征 + EGO-TOPO 图 + HOI
特征模型，结果必须带此限定。

## 2. 已有 Stage 1→Stage 2 生成式实验

已有链路预测 20 个未来动作序列，得到 next-action Top-k 和 normalized edit distance。
这些指标可作为本项目生成式消融/诊断，但不属于上述两个官方 EK55 表格协议：

- 五次采样并不等价于分类器按同一概率分布排序出的 Top-5。
- 876 条样本来自 8 步预测历史可用子集，而不是官方短期 anticipation validation split。
- GRPO checkpoint 的主要提升是输出格式，100 样本诊断中语义预测没有改善。

## 3. 可报告层级

1. `ek55_anticipation_topk_output/metrics.json`：只有在官方 split、目标动作前 1 秒输入、
   完整特征覆盖条件满足时，才可作为短期 Action Top-1/Top-5 主结果。
2. `ek55_future_verb_map_output/metrics.json`：可报告 P=25/50/75 All/Freq/Rare mAP，
   但当前版本必须标为 Stage-1 feature-limited baseline。
3. 原 3.56%/11.16% 与 Stage-2 0.114%/0.114%：仅作诊断，不与 DCR/CPM/INSIGHT
   表格横向比较。
