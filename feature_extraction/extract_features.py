"""
特征提取流水线: JPEG → HandObjDet + SAM2 + CLIP → .pt 文件

输出:
  frame_features_dir/{split}/{clip_uid}_{action_idx}.pt  (1024-dim)
  mask_features_dir/{split}/{clip_uid}_{action_idx}.pt   (1024-dim)

用法: python extract_features.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os, json, sys
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPVisionConfig

# ============================================================
# 配置
# ============================================================
DATA_DIR   = '/data/datasets/EPIC-KITCHENS'
ANN_DIR    = f'{DATA_DIR}/insight_annotations'
OUT_FRAME  = f'{DATA_DIR}/features/frame_features'
OUT_MASK   = f'{DATA_DIR}/features/mask_features'
SAMPLES_PER_ACTION = 4    # 每个 action 采样帧数
BATCH_SIZE = 32           # CLIP 批处理大小
DEVICE = 'cuda'
SKIP_EXISTING = True      # 跳过已有 .pt 文件

os.makedirs(f'{OUT_FRAME}/train', exist_ok=True)
os.makedirs(f'{OUT_FRAME}/val', exist_ok=True)
os.makedirs(f'{OUT_MASK}/train', exist_ok=True)
os.makedirs(f'{OUT_MASK}/val', exist_ok=True)

# ============================================================
# 1. 加载 CLIP ViT-L/14 (EgoVideo = CLIP)
# ============================================================
def build_clip():
    config = CLIPVisionConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        image_size=224, patch_size=14, intermediate_size=4096
    )
    model = CLIPVisionModel(config)

    ckpt = torch.load('/data/pretrain_model/EgoVideo/checkpoints/large_best.pt',
                      map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    clip_state = {}

    for k, v in state.items():
        if not k.startswith('module.visual.'):
            continue
        if any(x in k for x in ['temporal', 'image_projection', 'logit_scale']):
            continue

        # pos_embed 插值: [577,1024] → [257,1024]
        if 'positional_embedding' in k:
            old_size = int((v.shape[0] - 1) ** 0.5)
            new_size = 224 // 14
            cls_token = v[0:1]
            spatial = v[1:].reshape(old_size, old_size, 1024).permute(2, 0, 1).unsqueeze(0)
            spatial = F.interpolate(spatial, size=(new_size, new_size), mode='bicubic', align_corners=False)
            spatial = spatial.squeeze(0).permute(1, 2, 0).reshape(-1, 1024)
            clip_state['vision_model.embeddings.position_embedding.weight'] = \
                torch.cat([cls_token, spatial], dim=0)
            continue

        # qkv 拆分
        if 'attn.Wqkv' in k:
            layer_idx = k.split('.resblocks.')[1].split('.')[0]
            parts = v.chunk(3, dim=0)
            suffix = 'weight' if '.weight' in k else 'bias'
            clip_state[f'vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.{suffix}'] = parts[0]
            clip_state[f'vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.{suffix}'] = parts[1]
            clip_state[f'vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.{suffix}'] = parts[2]
            continue

        # 键名映射
        new_k = k.replace('module.visual.', 'vision_model.')
        new_k = new_k.replace('transformer.resblocks.', 'encoder.layers.')
        new_k = new_k.replace('ln_1.', 'layer_norm1.')
        new_k = new_k.replace('ln_2.', 'layer_norm2.')
        new_k = new_k.replace('ln_pre.', 'pre_layrnorm.')
        new_k = new_k.replace('ln_post.', 'post_layernorm.')
        new_k = new_k.replace('attn.out_proj.', 'self_attn.out_proj.')
        new_k = new_k.replace('mlp.fc1.', 'mlp.fc1.')
        new_k = new_k.replace('mlp.fc2.', 'mlp.fc2.')
        new_k = new_k.replace('conv1.', 'embeddings.patch_embedding.')
        new_k = new_k.replace('class_embedding', 'embeddings.class_embedding')

        # conv1: Conv3d [1024,3,1,14,14] → Conv2d [1024,3,14,14]
        if 'conv1.weight' in k:
            v = v.reshape(1024, 3, 1, 14, 14).squeeze(2)
        clip_state[new_k] = v

    model.load_state_dict(clip_state)
    model.to(DEVICE).eval()
    return model

# ============================================================
# 2. 加载 Hand Object Detector
# ============================================================
def build_handdet():
    sys.path.insert(0, '/data/pretrain_model/Hand_Object_Detector')
    from handobdet import HandObjectDetector
    return HandObjectDetector(input_format="BGR")  # cv2.imread 是 BGR

# ============================================================
# 3. 加载 SAM2
# ============================================================
def build_sam2():
    sys.path.insert(0, '/home/amax/ldy/git/sam2')
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2
    checkpoint = '/data/pretrain_model/SAM2/sam2.1_hiera_large.pt'
    model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    sam2 = build_sam2(model_cfg, checkpoint, device=DEVICE)
    return SAM2ImagePredictor(sam2)

# ============================================================
# 4. 读取帧
# ============================================================
def load_frame(participant, frame_idx):
    """读取单个 JPEG 帧, 返回 RGB numpy (H, W, 3)"""
    path = f'{DATA_DIR}/{participant}/rgb_frames/frame_{frame_idx:010d}.jpg'
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return img  # BGR

def load_frames_batch(frame_specs):
    """
    批量加载帧
    frame_specs: [(participant, frame_idx), ...]
    返回: list of RGB numpy arrays
    """
    imgs = []
    for p, f in frame_specs:
        img = load_frame(p, f)
        imgs.append(img)
    return imgs

# ============================================================
# 5. CLIP 预处理 + 批量编码
# ============================================================
def clip_preprocess(img_rgb):
    """单张 RGB 图像 → CLIP 输入 tensor [1,3,224,224]"""
    img = cv2.resize(img_rgb, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std  = np.array([0.26862954, 0.26130258, 0.27577711])
    img = (img - mean) / std
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)

def extract_frame_features(clip_model, frames_rgb):
    """
    frames_rgb: list of RGB numpy arrays
    返回: averaged [1024] feature
    """
    tensors = []
    for img in frames_rgb:
        if img is None:
            continue
        t = clip_preprocess(img)
        tensors.append(t)
    if not tensors:
        return None
    batch = torch.cat(tensors, dim=0)  # [N, 3, 224, 224]
    with torch.no_grad():
        feats = clip_model(batch).pooler_output  # [N, 1024]
    return feats.mean(dim=0)  # [1024]

def extract_mask_features(clip_model, frames_rgb, masks):
    """
    frames_rgb: list of RGB numpy arrays
    masks: list of binary masks (H, W)
    返回: averaged [1024] feature
    """
    tensors = []
    for img, mask in zip(frames_rgb, masks):
        if img is None or mask is None:
            continue
        # 应用 mask
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]))
        masked = img * mask[:,:,None].astype(np.float32)
        t = clip_preprocess(masked.astype(np.uint8))
        tensors.append(t)
    if not tensors:
        return None
    batch = torch.cat(tensors, dim=0)
    with torch.no_grad():
        feats = clip_model(batch).pooler_output
    return feats.mean(dim=0)  # [1024]

# ============================================================
# 6. HandObjDet + SAM2 → HOI mask
# ============================================================
def get_hand_mask(handdet, sam2_predictor, img_bgr):
    """
    输入: BGR 图像
    输出: hand mask (H, W) binary
    """
    try:
        hand_dets, obj_dets = handdet.detect(img_bgr, viz=False)
    except Exception:
        return None

    if hand_dets is None or len(hand_dets) == 0:
        return None

    # 取置信度最高的手 bbox
    best = hand_dets[hand_dets[:, 4].argmax().item()]
    x1, y1, x2, y2 = best[:4].cpu().numpy().astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None

    # SAM2 bbox prompt
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sam2_predictor.set_image(img_rgb)
    bbox = np.array([[x1, y1, x2, y2]])
    masks, scores, _ = sam2_predictor.predict(box=bbox, multimask_output=False)
    return masks[0]  # (H, W) binary

# ============================================================
# 7. 主流程
# ============================================================
def process_split(clip_model, handdet, sam2_pred, split):
    """
    处理一个 split (train/val) 的所有 actions
    """
    ann_file = f'{ANN_DIR}/{split}.json'
    if not os.path.exists(ann_file):
        print(f"⚠️ {ann_file} 不存在，跳过")
        return

    with open(ann_file) as f:
        data = json.load(f)

    clips = data['clips']
    # 按 clip_uid 分组
    by_clip = {}
    for c in clips:
        by_clip.setdefault(c['clip_uid'], []).append(c)

    # 也需要原始 CSV 来获取 start/stop frame
    # 重新读 CSV 建立索引
    frame_ranges = {}
    for csv_name in ['EPIC_train_action_labels.csv']:
        with open(f'{DATA_DIR}/annotations/{csv_name}') as f:
            for row in csv.DictReader(f):
                key = (row['video_id'], int(row['uid']))
                frame_ranges[key] = (int(row['start_frame']), int(row['stop_frame']))

    total = len(clips)
    done = 0
    skipped = 0

    pbar = tqdm(by_clip.items(), desc=f'[{split}]')
    for clip_uid, actions in pbar:
        participant = clip_uid.split('_')[0]  # P01_01 → P01

        for act in actions:
            key = (clip_uid, act['action_idx'])
            if key not in frame_ranges:
                continue

            start_f, stop_f = frame_ranges[key]
            n_frames = stop_f - start_f

            # 检查是否已存在
            f_out = f'{OUT_FRAME}/{split}/{clip_uid}_{act["action_idx"]}.pt'
            m_out = f'{OUT_MASK}/{split}/{clip_uid}_{act["action_idx"]}.pt'

            if SKIP_EXISTING and os.path.exists(f_out) and os.path.exists(m_out):
                skipped += 1
                continue

            # 采样帧
            if n_frames <= SAMPLES_PER_ACTION:
                sample_idxs = list(range(start_f, stop_f))
            else:
                step = n_frames / SAMPLES_PER_ACTION
                sample_idxs = [int(start_f + i * step) for i in range(SAMPLES_PER_ACTION)]

            frames_rgb = []
            for fi in sample_idxs:
                img = load_frame(participant, fi)
                if img is not None:
                    frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if len(frames_rgb) < 2:  # 至少需要 2 帧
                continue

            # 帧特征
            frame_feat = extract_frame_features(clip_model, frames_rgb)
            if frame_feat is not None:
                torch.save(frame_feat.cpu(), f_out)

            # HOI 特征 (sam2_predictor 每帧需要 set_image，这里简化处理，取第一帧的 mask)
            # 实际完整版应该每帧分别处理
            # 这里先只用第一帧做 mask，后面可以优化
            first_img_bgr = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2BGR)
            mask = get_hand_mask(handdet, sam2_pred, first_img_bgr)

            if mask is not None:
                # 用同一 mask 对所有帧提 HOI 特征（简化版）
                masks_repeated = [mask] * len(frames_rgb)
                mask_feat = extract_mask_features(clip_model, frames_rgb, masks_repeated)
                if mask_feat is not None:
                    torch.save(mask_feat.cpu(), m_out)

            done += 1
            pbar.set_postfix({'done': done, 'skip': skipped})

    print(f"[{split}] 完成: {done} 个动作, 跳过 {skipped} 个")

# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    print("加载 CLIP...")
    clip = build_clip()
    print("加载 HandObjDet...")
    handdet = build_handdet()
    print("加载 SAM2...")
    sam2 = build_sam2()

    for split in ['train', 'val']:
        process_split(clip, handdet, sam2, split)

    print("✅ 特征提取完成!")
