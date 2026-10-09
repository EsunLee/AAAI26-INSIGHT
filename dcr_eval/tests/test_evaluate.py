from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "04_evaluate.py"
SPEC = importlib.util.spec_from_file_location("dcr_evaluate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateTop1Test(unittest.TestCase):
    def test_parser_normalizes_case_and_whitespace(self):
        answer = MODULE.extract_answer("<ANSWER>  Take   Cup, open fridge </ANSWER>")
        self.assertEqual(MODULE.first_action(answer), "take cup")

    def test_end_to_end_top1_and_parse_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            pred_dir = root / "predictions"
            output_path = root / "metrics.json"
            pred_dir.mkdir()

            records = [
                {"clip_uid": "P01_01", "action_idx": 1, "gt_action_text": "take cup"},
                {"clip_uid": "P01_01", "action_idx": 2, "gt_action_text": "open fridge"},
            ]
            predictions = [
                {"response": "<answer>Take   Cup, close door</answer>"},
                {"response": None},
            ]
            input_path.write_text("".join(json.dumps(x) + "\n" for x in records))
            (pred_dir / "candidate_0.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in predictions)
            )

            argv = [
                "04_evaluate.py", "--input", str(input_path),
                "--predictions-dir", str(pred_dir), "--output", str(output_path),
            ]
            with patch.object(sys, "argv", argv):
                MODULE.main()

            metrics = json.loads(output_path.read_text())
            self.assertEqual(metrics["samples_labeled"], 2)
            self.assertEqual(metrics["top1_correct"], 1)
            self.assertEqual(metrics["next_action_top1"], 50.0)
            self.assertEqual(metrics["parse_failures"], 1)
            self.assertEqual(metrics["parse_success_rate"], 50.0)


if __name__ == "__main__":
    unittest.main()
