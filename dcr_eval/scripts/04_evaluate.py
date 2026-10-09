"""04 — 评测：Stage 2 Top-1 预测 vs GT → Next-Action Exact-Match@1。

输入（全部可参数化，支持 smoke 与全量分别评测）：
  --input              stage2_input.jsonl（含 gt_action_text / clip_uid / action_idx）
  --predictions-dir    含 candidate_0.jsonl 的目录（三 GPU 分片需先 cat 合并）
  --output             metrics.json 输出路径

判定（Top-1 口径）：
  - candidate_0 的 answer 第一个动作 == GT 动作文本（精确匹配，大小写不敏感、空白规范化）
  - 解析失败（无 <answer> 或空内容）计入 parse_failures，不计命中

对齐：
  - 候选行数 == 输入行数（严格校验，禁止静默错位）
  - 候选含 clip_uid/action_idx 时按样本 ID 重排（并要求 ID 无重复），否则按行号（并警告）
  - 有/无标签混合时：先按原始顺序配对，再共同过滤有 GT 的记录（避免错位）

运行：python scripts/04_evaluate.py --input <...> --predictions-dir <...> --output <...>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)  # 大小写不敏感


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=C.PRED_DIR / "stage2_input.jsonl",
        help="输入记录（默认 config 的 stage2_input.jsonl；smoke 时传 /tmp/stage2_input_100.jsonl）",
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=C.GEN_DIR,
        help="含 candidate_0.jsonl 的目录（默认 config 的 stage2_generated）",
    )
    parser.add_argument(
        "--output", type=Path, default=C.EVAL_DIR / "metrics.json",
        help="指标输出路径（默认 config 的 evaluation/metrics.json）",
    )
    return parser.parse_args()


def extract_answer(response: str) -> str | None:
    """从模型输出中提取 <answer> 内容；缺失时返回 None。"""
    m = ANSWER_RE.search(response)
    return m.group(1).strip() if m else None


def first_action(answer: str) -> str | None:
    """answer 内容按逗号切分，取第一个动作，规范化（小写、连续空白合并）。"""
    if not answer:
        return None
    first = answer.split(",")[0].strip().lower()
    first = re.sub(r"\s+", " ", first)  # "take   cup" → "take cup"
    return first if first else None


def load_input_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: JSON 损坏: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: 期望 JSON object")
                records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_top1_responses(gen_dir: Path) -> tuple[list[str], list[tuple | None]]:
    """读 candidate_0.jsonl，返回 (responses, sample_ids)。"""
    path = gen_dir / "candidate_0.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}（先运行 03_stage2_generate.sh 并合并分片）")
    responses, ids = [], []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 损坏: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number}: 期望 JSON object")
            if "response" in obj:
                response = obj["response"]
            elif "choices" in obj:
                response = obj["choices"][0]["message"]["content"]
            elif "generated_text" in obj:
                response = obj["generated_text"]
            else:
                response = obj.get("output", "")
            responses.append(response if isinstance(response, str) else "")
            cid = obj.get("clip_uid")
            aidx = obj.get("action_idx")
            ids.append((str(cid), int(aidx)) if cid is not None and aidx is not None else None)
    return responses, ids


def main() -> None:
    args = parse_args()
    records = load_input_records(args.input)
    responses, cand_ids = load_top1_responses(args.predictions_dir)

    # 严格行数校验
    if len(responses) != len(records):
        raise ValueError(
            f"候选行数 {len(responses)} != 输入行数 {len(records)}："
            "swift 输出可能缺失样本，请检查日志后重跑 03"
        )

    # 尽量按样本 ID 对齐（不依赖"输出顺序 == 输入顺序"）
    has_ids = [sid is not None for sid in cand_ids]
    if any(has_ids) and not all(has_ids):
        raise ValueError("候选记录只有部分包含 clip_uid/action_idx，无法可靠对齐")
    if all(has_ids):
        input_ids = [f"{r['clip_uid']}_{r['action_idx']}" for r in records]
        cand_ids_s = [f"{cid}_{aidx}" for cid, aidx in cand_ids]
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("输入样本 ID 存在重复，禁止对齐")
        if len(set(cand_ids_s)) != len(cand_ids_s):
            raise ValueError("候选样本 ID 存在重复，禁止对齐")
        if set(cand_ids_s) != set(input_ids):
            raise ValueError(
                "候选样本 ID 集合与输入不一致（缺失或多余样本），禁止按行号对齐"
            )
        if cand_ids_s != input_ids:
            pos = {cid: idx for idx, cid in enumerate(cand_ids_s)}
            responses = [responses[pos[iid]] for iid in input_ids]
            print("[align] 候选顺序与输入不同，已按 (clip_uid, action_idx) 重排")
        else:
            print("[align] 候选顺序 == 输入顺序（已校验 ID 集合一致且无重复）")
    else:
        print("[warn] 候选文件不含 clip_uid/action_idx，按行号对齐（依赖 swift 保持输入顺序）")

    # 先按原始顺序配对 (record, response)，再共同过滤有 GT 的记录（避免错位）
    pairs = list(zip(records, responses))
    labeled_pairs = [(r, resp) for r, resp in pairs if r.get("gt_action_text")]
    n_total = len(records)
    n_labeled = len(labeled_pairs)
    print(f"[samples] total={n_total} labeled(有 GT)={n_labeled}")
    if n_labeled == 0:
        print("无标签样本（challenge/test），无法计算 Top-1，仅输出预测文件。")
        return

    hits = parse_fail = 0
    for rec, resp in labeled_pairs:
        gt = re.sub(r"\s+", " ", rec["gt_action_text"].lower().strip())
        answer = extract_answer(resp)
        first = first_action(answer)
        if first is None:
            parse_fail += 1
            continue
        if first == gt:
            hits += 1

    parse_successes = n_labeled - parse_fail
    metrics = {
        "protocol": "INSIGHT Next-Action Exact-Match@1 (DCR-valid, generative)",
        "metric_definition": "candidate_0 (temperature 0) first action == GT action text",
        "run_name": C.RUN_NAME,
        "input": str(args.input),
        "predictions": str(args.predictions_dir / "candidate_0.jsonl"),
        "input_sha256": sha256_file(args.input),
        "predictions_sha256": sha256_file(args.predictions_dir / "candidate_0.jsonl"),
        "samples_labeled": n_labeled,
        "top1_correct": hits,
        "next_action_top1": round(100.0 * hits / n_labeled, 3),
        "parse_failures": parse_fail,
        "parse_successes": parse_successes,
        "parse_success_rate": round(100.0 * parse_successes / n_labeled, 3),
        "caveats": [
            "recognition-gated history + generative prediction; not comparable to "
            "DCR official tau_a=1s Top-1 (19.2 on EK55 valid)",
            "source frame path omits video_id and can be ambiguous for participants "
            "with multiple videos",
            "Stage1 mask branch reuses RGB frame feature because true HOI features are unavailable",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
