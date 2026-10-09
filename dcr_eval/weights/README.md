# 权重清单（本项目不内置权重，运行时指向远程路径）

运行前把 [config.py](../config.py) 中的权重路径与实际位置对齐。

| 权重 | 远程路径（默认） | 用途 |
|------|------|------|
| CLIP ViT-L/14 | `/data/pretrain_model/EgoVideo/checkpoints/large_best.pt` | 帧特征提取（1024 维） |
| Stage 1 模型 | `/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth` | 动作识别（verb/noun） |
| 共现矩阵 | `/data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt` | 公式5 语义先验重排 |
| Qwen2.5-VL-7B | `/data/pretrain_model/Qwen2.5-VL-7B-Instruct` | Stage 2 基座 |
| SFT LoRA adapter | `/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/checkpoint-200` | Stage 2 微调权重（**最佳 checkpoint-200，非 895**） |

## 校验

```bash
# 全部存在则输出 OK
for p in \
  /data/pretrain_model/EgoVideo/checkpoints/large_best.pt \
  /data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth \
  /data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt \
  /data/pretrain_model/Qwen2.5-VL-7B-Instruct \
  /data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/checkpoint-200; do
  [ -e "$p" ] && echo "OK  $p" || echo "MISS $p"
done
```
