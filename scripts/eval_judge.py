#!/usr/bin/env python3
"""
LLM-as-judge evaluation using Nemotron-3-Super (served locally on port 8123).

Reads one or more responses JSONL files (from generate_responses.py), asks Super
to score each (prompt, response, ground_truth) triple 1-5 on factuality and
relevance, and writes the aggregated results to disk.

Also computes ROUGE-L F1 against the ground truth as a reference-free number.

Usage (Super must be up on port 8123):
  python3 scripts/eval_judge.py \\
    --responses outputs/comparison/responses/*.jsonl \\
    --out       outputs/comparison/judge_results.jsonl \\
    --summary   outputs/comparison/judge_summary.json \\
    --base-url  http://localhost:8123/v1
"""
import argparse
import glob
import json
import random
import re
import time
from itertools import combinations
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from rouge_score import rouge_scorer


JUDGE_SYSTEM = (
    "You are an expert evaluator of AI assistants. "
    "You score answers on factuality and relevance compared to a reference answer. "
    "Be strict but fair. Always respond in the exact format requested."
)

JUDGE_USER = """\
Evaluate the AI ASSISTANT RESPONSE against the REFERENCE ANSWER on a scale of 1 to 5:
  1 = Completely wrong or irrelevant
  2 = Mostly wrong or off-topic
  3 = Partially correct but missing key info, or contains notable errors
  4 = Largely correct with minor issues
  5 = Matches the reference in quality and correctness

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

AI ASSISTANT RESPONSE:
{response}

Respond with ONLY a single digit (1, 2, 3, 4, or 5) as the first character of your reply.
Then optionally one short sentence of justification.
"""


def parse_score(text: str) -> int | None:
    """Extract an unambiguous score without mistaking prose digits for it."""
    if not text:
        return None
    # The prompt requires the first character to be the score.
    m = re.match(r"\s*([1-5])\b", text)
    if m:
        return int(m.group(1))
    # Reasoning servers sometimes expose only the reasoning field. In that
    # case accept an explicit score label near the end, but never an arbitrary
    # digit from the justification.
    m = re.search(
        r"(?:final\s+)?(?:score|rating|answer|grade)[:\s]+([1-5])\b",
        text[-300:],
        re.IGNORECASE,
    )
    return int(m.group(1)) if m else None


def bootstrap_delta_ci(
    pairs: list[tuple[int, int]], seed: int, samples: int = 10_000
) -> list[float] | None:
    """Paired bootstrap 95% CI for method score minus base score."""
    if not pairs:
        return None
    rng = random.Random(seed)
    deltas = [method - base for base, method in pairs]
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    means.sort()
    return [means[int(samples * 0.025)], means[int(samples * 0.975)]]


def build_summary(per_method: dict[str, list[dict]], seed: int) -> dict:
    methods: dict[str, dict] = {}
    judged_prompt_sets = []
    for method, recs in per_method.items():
        scores = [r["judge_score"] for r in recs if r["judge_score"] is not None]
        rouges = [r["rouge_l"] for r in recs]
        judged_prompt_sets.append(
            {r["prompt"] for r in recs if r["judge_score"] is not None}
        )
        methods[method] = {
            "n": len(recs),
            "n_judged": len(scores),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "mean_rouge": (sum(rouges) / len(rouges)) if rouges else None,
            "pct_5": (sum(s == 5 for s in scores) / len(scores) * 100) if scores else None,
            "pct_ge_4": (sum(s >= 4 for s in scores) / len(scores) * 100) if scores else None,
        }

    common_prompts = set.intersection(*judged_prompt_sets) if judged_prompt_sets else set()
    by_method = {
        method: {r["prompt"]: r for r in recs}
        for method, recs in per_method.items()
    }
    for method, stats in methods.items():
        common_scores = [by_method[method][p]["judge_score"] for p in common_prompts]
        stats["common_n"] = len(common_scores)
        stats["common_mean_score"] = (
            sum(common_scores) / len(common_scores) if common_scores else None
        )

    if "base" in by_method:
        base = by_method["base"]
        for method, records in by_method.items():
            if method == "base":
                continue
            shared = sorted(base.keys() & records.keys())
            pairs = [
                (base[p]["judge_score"], records[p]["judge_score"])
                for p in shared
                if base[p]["judge_score"] is not None
                and records[p]["judge_score"] is not None
            ]
            wins = sum(method_score > base_score for base_score, method_score in pairs)
            ties = sum(method_score == base_score for base_score, method_score in pairs)
            losses = len(pairs) - wins - ties
            deltas = [method_score - base_score for base_score, method_score in pairs]
            methods[method]["paired_vs_base"] = {
                "n": len(pairs),
                "method_mean": (
                    sum(method_score for _, method_score in pairs) / len(pairs)
                    if pairs else None
                ),
                "base_mean": (
                    sum(base_score for base_score, _ in pairs) / len(pairs)
                    if pairs else None
                ),
                "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "delta_ci_95": bootstrap_delta_ci(pairs, seed),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / len(pairs) * 100 if pairs else None,
                "win_rate_excluding_ties": (
                    wins / (wins + losses) * 100 if wins + losses else None
                ),
            }

    pairwise = []
    for left, right in combinations(sorted(by_method), 2):
        shared = sorted(by_method[left].keys() & by_method[right].keys())
        pairs = [
            (
                by_method[left][p]["judge_score"],
                by_method[right][p]["judge_score"],
            )
            for p in shared
            if by_method[left][p]["judge_score"] is not None
            and by_method[right][p]["judge_score"] is not None
        ]
        right_wins = sum(right_score > left_score for left_score, right_score in pairs)
        ties = sum(right_score == left_score for left_score, right_score in pairs)
        left_wins = len(pairs) - right_wins - ties
        deltas = [right_score - left_score for left_score, right_score in pairs]
        pairwise.append({
            "left": left,
            "right": right,
            "n": len(pairs),
            "mean_delta_right_minus_left": (
                sum(deltas) / len(deltas) if deltas else None
            ),
            "delta_ci_95": bootstrap_delta_ci(pairs, seed),
            "right_wins": right_wins,
            "ties": ties,
            "left_wins": left_wins,
        })

    missing = sum(
        r["judge_score"] is None for recs in per_method.values() for r in recs
    )
    return {
        "complete": missing == 0,
        "missing_judgments": missing,
        "common_judged_n": len(common_prompts),
        "methods": methods,
        "pairwise": pairwise,
    }


def judge_once(client, model_id: str, judge_prompt: str, max_retries: int, delay: float):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            reasoning = (getattr(msg, "reasoning_content", "") or "").strip()
            raw = content or reasoning
            score = parse_score(content) or parse_score(reasoning)
            if score is not None:
                return score, raw, None
            last_error = "judge returned no unambiguous score"
        except Exception as exc:
            last_error = str(exc)
            raw = ""
        if attempt < max_retries:
            time.sleep(delay * (attempt + 1))
    return None, raw, last_error


def main(args):
    # Expand glob patterns
    files: list[Path] = []
    for pattern in args.responses:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"WARNING: no files match {pattern}")
            continue
        files.extend(Path(p) for p in expanded)
    files = sorted(set(files))
    print(f"Judging {len(files)} response files: {[f.name for f in files]}")

    client = OpenAI(base_url=args.base_url, api_key="local")
    model_id = client.models.list().data[0].id
    print(f"Judge model: {model_id}")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse successful prior judgments. This makes retries cheap after a server
    # interruption while ensuring changed responses are judged again.
    reusable: dict[tuple[str, str], dict] = {}
    for candidate in (out_path, out_path.with_suffix(out_path.suffix + ".partial")):
        if not candidate.exists():
            continue
        for line in open(candidate):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("judge_score") is not None:
                reusable[(rec["method"], rec["prompt"])] = rec

    per_method: dict[str, list[dict]] = defaultdict(list)
    total_processed = 0
    total_reused = 0
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")

    with open(partial_path, "w") as out_f:
        for path in files:
            records = [json.loads(l) for l in open(path) if l.strip()]
            print(f"\n=== {path.name} ({len(records)} records) ===")
            for i, rec in enumerate(records):
                prompt = rec["prompt"]
                response = rec.get("response", "")
                gt = rec.get("ground_truth", "")
                method = rec.get("model", path.stem)

                # ROUGE-L (no judge needed)
                try:
                    rouge_l = scorer.score(gt, response)["rougeL"].fmeasure if gt and response else 0.0
                except Exception:
                    rouge_l = 0.0

                prior = reusable.get((method, prompt))
                if (
                    prior
                    and prior.get("response") == response
                    and prior.get("ground_truth") == gt
                ):
                    record = prior
                    record["rouge_l"] = rouge_l
                    total_reused += 1
                    elapsed = 0.0
                else:
                    # LLM-as-judge
                    judge_prompt = JUDGE_USER.format(
                        question=prompt, reference=gt, response=response
                    )
                    t0 = time.perf_counter()
                    score, raw, error = judge_once(
                        client, model_id, judge_prompt, args.max_retries, args.retry_delay
                    )
                    elapsed = time.perf_counter() - t0
                    record = {
                        "method": method,
                        "prompt": prompt,
                        "response": response,
                        "ground_truth": gt,
                        "judge_score": score,
                        "judge_raw": raw,
                        "judge_error": error,
                        "rouge_l": rouge_l,
                    }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                per_method[method].append(record)
                total_processed += 1

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{len(records)}] {elapsed:.1f}s score={record['judge_score']} rougeL={rouge_l:.3f}",
                          flush=True)

    partial_path.replace(out_path)
    summary = build_summary(per_method, args.seed)
    summary["protocol"] = args.protocol

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for method, stats in summary["methods"].items():
        print(f"  {method}: mean_score={stats['mean_score']}  "
              f"common_mean={stats['common_mean_score']}  "
              f"mean_rouge={stats['mean_rouge']}  "
              f"judged={stats['n_judged']}/{stats['n']}")
    print(f"\nWrote {total_processed} judge records to {out_path}")
    print(f"Reused {total_reused} successful prior judgments")
    print(f"Wrote summary to {summary_path}")
    if not summary["complete"] and not args.allow_missing:
        raise SystemExit(
            f"{summary['missing_judgments']} judgments remain missing; rerun to retry "
            "or pass --allow-missing to accept an incomplete comparison"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", nargs="+", required=True,
                        help="Response JSONL files or glob patterns")
    parser.add_argument("--out",       default="outputs/comparison/judge_results.jsonl")
    parser.add_argument("--summary",   default="outputs/comparison/judge_summary.json")
    parser.add_argument("--base-url",  default="http://localhost:8123/v1")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocol", default="source_grouped_test",
        help="Protocol label recorded in the summary for provenance",
    )
    parser.add_argument("--allow-missing", action="store_true")
    main(parser.parse_args())
