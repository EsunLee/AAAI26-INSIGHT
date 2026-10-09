"""DCR EK55 标注的严格、共享读取函数。"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


VIDEO_ID_RE = re.compile(r"^P\d{2}_\d{2,3}$")
REQUIRED_ACTION_COLUMNS = {
    "uid", "video_id", "participant_id", "start_timestamp",
    "start_frame", "stop_frame", "verb_class", "noun_class",
}


def load_video_ids(path: Path) -> set[str]:
    """读取单列视频划分，同时兼容有表头和无表头 CSV。"""
    raw = pd.read_csv(path, header=None, dtype=str)
    if raw.shape[1] < 1:
        raise ValueError(f"{path}: 没有可读列")
    values = {value.strip() for value in raw.iloc[:, 0].dropna() if value.strip()}
    values -= {"video_id", "video", "videos"}
    invalid = sorted(value for value in values if not VIDEO_ID_RE.fullmatch(value))
    if invalid:
        raise ValueError(f"{path}: 非法视频 ID（前 10 个）: {invalid[:10]}")
    if not values:
        raise ValueError(f"{path}: 视频列表为空")
    return values


def load_action_labels(path: Path) -> pd.DataFrame:
    """读取 DCR EK55 action pickle，将 UID 索引统一为普通列并做结构校验。"""
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path}: 期望 pandas.DataFrame，实际={type(df).__name__}")
    if "uid" not in df.columns:
        index_name = df.index.name
        df = df.reset_index()
        generated_name = index_name if index_name is not None else "index"
        if generated_name not in df.columns:
            raise ValueError(f"{path}: 重置索引后找不到 UID 列")
        df = df.rename(columns={generated_name: "uid"})
    missing = sorted(REQUIRED_ACTION_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path}: 缺少必需列 {missing}; actual={list(df.columns)}")
    if df["uid"].isna().any() or not df["uid"].is_unique:
        raise ValueError(f"{path}: uid 存在空值或重复")
    for column in ("uid", "start_frame", "stop_frame", "verb_class", "noun_class"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
    bad_frame = df[df["stop_frame"] <= df["start_frame"]]
    if not bad_frame.empty:
        raise ValueError(f"{path}: 存在 stop_frame <= start_frame，数量={len(bad_frame)}")
    if not df["verb_class"].between(0, 124).all():
        raise ValueError(f"{path}: verb_class 超出 0..124")
    if not df["noun_class"].between(0, 351).all():
        raise ValueError(f"{path}: noun_class 超出 0..351")
    return df
