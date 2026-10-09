# 数据集说明（本项目不内置数据，运行时指向远程路径）

本项目评测目标是 **DCR 项目（CVPR 2022）的 EK55 数据**。真实数据在远程服务器，
运行前把 [config.py](../config.py) 中的路径与实际部署位置对齐。

## 需要的数据（DCR 项目自带，`/data/sp/projects/csj/DCR-main/data/EK55/`）

| 文件 | 作用 | 状态 |
|------|------|------|
| `EPIC_train_action_labels.pkl` | EK55 训练标注（28472 条，含 verb/noun 类、起止帧） | DCR 自带 |
| `validation_videos.csv` | **官方 valid 划分**（有标签，用于自评 Top-1） | DCR 自带 |
| `training_videos.csv` | 官方 train 划分 | DCR 自带 |
| `EPIC_test_s1/s2_timestamps.pkl` | 官方 test 划分（**无标签**，challenge；只能提交官方评测） | DCR 自带 |
| `EPIC_verb_classes.csv` / `EPIC_noun_classes.csv` | 类 id → 名称映射 | DCR 自带 |
| `data/feature/EK55_RGB_TSM/data.mdb` | RULSTM TSM 特征（2048 维 8fps，19.3GB） | DCR 自带（**本项目不使用**，见下） |

## 需要的帧（我们自己的 EK55 数据，`/data/datasets/EPIC-KITCHENS/`）

```
{Participant}/rgb_frames/frame_{帧号:010d}.jpg    （60fps JPEG，125 万张）
```

DCR valid/test 的 28 个 participant 帧目录已确认全部存在。

## 为什么不用 DCR 的 TSM 特征

我们的 Stage 1 权重是在 **CLIP 1024 维特征**上训练的；DCR 的 TSM 特征（2048 维）
与 CLIP 分布不同，直接喂权重会因分布漂移导致效果崩溃。因此本项目
**用我们的 CLIP 管线从原始帧重新提取特征**（01 脚本），TSM mdb 仅作记录。

## 脚本运行后生成

| 路径 | 内容 |
|------|------|
| `features/val/{video}_{uid}.pt` | 每段 4 帧平均的 CLIP 1024 维 RGB 特征（复刻 Stage1 训练提取器；mask = 帧副本） |
| `annotations/val.json` | insight 格式标注（clips: clip_uid/action_idx/verb_label/noun_label） |
| `annotations/meta.json` | 统计（条数、覆盖率） |
