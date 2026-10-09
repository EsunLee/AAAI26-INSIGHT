"""
将 EK55 CSV 标注转换为 INSIGHT Stage 1 需要的 JSON 格式。
输出: annotation_dir/train.json, val.json
格式: {"clips": [{"clip_uid": "P01_01", "action_idx": 0, "verb_label": 2, "noun_label": 8}, ...]}
"""

import csv, json, os, random

ANN_DIR = '/data/datasets/EPIC-KITCHENS/annotations'
OUT_DIR  = '/data/datasets/EPIC-KITCHENS/insight_annotations'
VAL_RATIO = 0.1  # 10% 做验证集

os.makedirs(OUT_DIR, exist_ok=True)

def build_json(csv_file, output_json):
    clips = []
    clip_groups = {}

    with open(f'{ANN_DIR}/{csv_file}') as f:
        for row in csv.DictReader(f):
            clip_uid = row['video_id']  # e.g. P01_01
            entry = {
                'clip_uid': clip_uid,
                'action_idx': int(row['uid']),
                'verb_label': int(row['verb_class']),
                'noun_label': int(row['noun_class'])
            }
            clips.append(entry)
            clip_groups.setdefault(clip_uid, []).append(entry)

    # 按 clip 分 train/val
    all_clips = list(clip_groups.keys())
    random.shuffle(all_clips)
    n_val = int(len(all_clips) * VAL_RATIO)
    val_clips = set(all_clips[:n_val])
    train_clips = set(all_clips[n_val:])

    train_data = {'clips': [e for e in clips if e['clip_uid'] in train_clips]}
    val_data   = {'clips': [e for e in clips if e['clip_uid'] in val_clips]}

    with open(f'{OUT_DIR}/train.json', 'w') as f:
        json.dump(train_data, f)
    with open(f'{OUT_DIR}/val.json', 'w') as f:
        json.dump(val_data, f)

    print(f"{csv_file}: {len(clips)} actions, {len(clip_groups)} clips")
    print(f"  train: {len(train_data['clips'])} actions, {len(train_clips)} clips")
    print(f"  val:   {len(val_data['clips'])} actions, {len(val_clips)} clips")

build_json('EPIC_train_action_labels.csv', 'train.json')

# 统计 verb/noun 类数
verbs = set()
nouns = set()
for clip in json.load(open(f'{OUT_DIR}/train.json'))['clips']:
    verbs.add(clip['verb_label'])
    nouns.add(clip['noun_label'])

print(f"\nverb 类: {min(verbs)}-{max(verbs)} ({len(verbs)} 类)")
print(f"noun 类: {min(nouns)}-{max(nouns)} ({len(nouns)} 类)")
print(f"\n→ 需要修改 HandObject/main.py:")
print(f"   verb_num_classes = {max(verbs)+1}")
print(f"   noun_num_classes = {max(nouns)+1}")
print(f"✅ 标注转换完成！")
