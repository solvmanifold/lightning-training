import sys
import types
import unittest
from pathlib import Path

from data.prepare import deduplicate, grouped_split

# eval_judge's pure metric helpers do not need these optional runtime clients.
if "rouge_score" not in sys.modules:
    rouge_module = types.ModuleType("rouge_score")
    rouge_module.rouge_scorer = object()
    sys.modules["rouge_score"] = rouge_module

from scripts.eval_judge import build_summary, parse_score
from scripts.run_sweep import deep_merge, load_jobs


def record(prompt, answer, source=None):
    value = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }
    if source:
        value["metadata"] = {"source_file": source}
    return value


class SplitTests(unittest.TestCase):
    def test_deduplicate_removes_exact_and_conflicting_prompts(self):
        rows = [
            record("same", "answer", "a.py"),
            record("same", "answer", "a.py"),
            record("ambiguous", "one", "b.py"),
            record("ambiguous", "two", "c.py"),
            record("keep", "three", "d.py"),
        ]
        kept, exact, conflicts = deduplicate(rows)
        self.assertEqual([r["messages"][0]["content"] for r in kept], ["same", "keep"])
        self.assertEqual(exact, 1)
        self.assertEqual(conflicts, 2)

    def test_grouped_split_never_splits_a_source(self):
        rows = [
            record(f"q-{source}-{i}", "a", source)
            for source in ("a.py", "b.py", "c.py", "d.py", "e.py")
            for i in range(2)
        ]
        train, val, test = grouped_split(rows, 0.2, 0.2, seed=7)
        source_sets = [
            {r["metadata"]["source_file"] for r in split}
            for split in (train, val, test)
        ]
        self.assertFalse(source_sets[0] & source_sets[1])
        self.assertFalse(source_sets[0] & source_sets[2])
        self.assertFalse(source_sets[1] & source_sets[2])


class EvaluationTests(unittest.TestCase):
    def test_score_parser_rejects_digits_in_justification(self):
        self.assertEqual(parse_score("4 Minor issue"), 4)
        self.assertEqual(parse_score("Final score: 3"), 3)
        self.assertIsNone(parse_score("It misses 2 details"))

    def test_summary_uses_common_and_pairwise_prompts(self):
        def judged(method, prompt, score):
            return {
                "method": method,
                "prompt": prompt,
                "judge_score": score,
                "rouge_l": 0.1,
            }

        rows = {
            "base": [judged("base", "a", 1), judged("base", "b", 2)],
            "sft": [judged("sft", "a", 3), judged("sft", "b", None)],
            "dpo": [judged("dpo", "a", 3), judged("dpo", "b", 4)],
        }
        summary = build_summary(rows, seed=42)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["common_judged_n"], 1)
        self.assertEqual(summary["methods"]["sft"]["common_mean_score"], 3)
        paired = summary["methods"]["dpo"]["paired_vs_base"]
        self.assertEqual(paired["n"], 2)
        self.assertEqual(paired["mean_delta"], 2)
        dpo_sft = next(
            comparison for comparison in summary["pairwise"]
            if comparison["left"] == "dpo" and comparison["right"] == "sft"
        )
        self.assertEqual(dpo_sft["n"], 1)
        self.assertEqual(dpo_sft["mean_delta_right_minus_left"], 0)


class SweepTests(unittest.TestCase):
    def test_deep_merge_preserves_unmodified_nested_values(self):
        merged = deep_merge(
            {"training": {"lr": 1e-4, "steps": 10}, "rank": 4},
            {"training": {"steps": 3}},
        )
        self.assertEqual(merged["training"], {"lr": 1e-4, "steps": 3})
        self.assertEqual(merged["rank"], 4)

    def test_manifest_applies_tier_and_experiment_overrides(self):
        root = Path(__file__).resolve().parents[1]
        jobs = load_jobs(root / "experiments/sweep.yaml", "compare", ["r8-attn-moe"])
        self.assertEqual(len(jobs), 1)
        config = jobs[0]["config"]
        self.assertEqual(config["training"]["token_budget"], 250000)
        self.assertEqual(config["training"]["scheduler_steps"], 1500)
        self.assertEqual(config["lora"]["r"], 8)
        self.assertIn("up_proj", config["lora"]["target_modules"])


if __name__ == "__main__":
    unittest.main()