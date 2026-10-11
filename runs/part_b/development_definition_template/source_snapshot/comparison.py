"""Offline fairness scores, paired changes, and measured inference cost."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from a2.analysis import classification_metrics, score_response
from a2.experiment import validate_existing
from a2.io import read_jsonl, sha256, write_json
from a2.tasks import CLASS_LABELS


def scored_fairness(run_dir: Path) -> tuple[list[dict], list[dict]]:
    samples = json.loads((run_dir / "samples.json").read_text())
    all_records = read_jsonl(run_dir / "responses.jsonl")
    if validate_existing(all_records, samples) != {row["id"] for row in samples}:
        raise ValueError("Generation is incomplete.")
    records = {row["sample_id"]: row for row in all_records}
    fairness = [row for row in samples if row["dimension"] == "fairness"]
    if not fairness:
        raise ValueError("No fairness samples found.")
    selected = [records[row["id"]] for row in fairness]
    return [
        score_response(sample, record) for sample, record in zip(fairness, selected, strict=True)
    ], selected


def inference_cost(records: list[dict]) -> dict:
    durations = [row["elapsed_seconds"] for row in records]
    return {
        "requests": len(records),
        "retried_requests": sum(len(row["attempts"]) > 1 for row in records),
        "prompt_tokens": sum(row["response"]["usage"]["prompt_tokens"] for row in records),
        "completion_tokens": sum(row["response"]["usage"]["completion_tokens"] for row in records),
        "total_seconds": sum(durations),
        "mean_seconds": mean(durations),
        "median_seconds": median(durations),
    }


def fairness_summary(rows: list[dict], records: list[dict]) -> dict:
    labels = CLASS_LABELS["stereotype_recognition"]
    return {
        "metrics": classification_metrics(rows, labels),
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
        "responses_with_reasoning_content": sum(row["has_reasoning_content"] for row in rows),
        "cost": inference_cost(records),
        "by_topic": {
            topic: {
                "inputs": len(subset),
                "accuracy": sum(row["correct"] for row in subset) / len(subset),
            }
            for topic in sorted({row["metadata"]["type"] for row in rows})
            if (subset := [row for row in rows if row["metadata"]["type"] == topic])
        },
    }


def score_fairness(run_dir: Path) -> dict:
    rows, records = scored_fairness(run_dir)
    summary = {
        **fairness_summary(rows, records),
        "responses_sha256": sha256(run_dir / "responses.jsonl"),
    }
    write_json(run_dir / "scores.json", summary)
    write_json(run_dir / "scored_samples.json", rows)
    return summary


def paired_changes(baseline: list[dict], mitigated: list[dict]) -> dict:
    old, new = ({row["id"]: row for row in rows} for rows in (baseline, mitigated))
    if len(old) != len(baseline) or len(new) != len(mitigated) or old.keys() != new.keys():
        raise ValueError("Comparison requires the same unique sample IDs.")
    transitions = Counter()
    label_changes = 0
    for sample_id, before in old.items():
        after = new[sample_id]
        if any(before[key] != after[key] for key in ("prompt", "gold_label", "task")):
            raise ValueError("Comparison samples or reference labels changed.")
        transitions[
            f"{'correct' if before['correct'] else 'wrong'}_to_"
            f"{'correct' if after['correct'] else 'wrong'}"
        ] += 1
        label_changes += before["prediction"] != after["prediction"]
    return {
        "inputs": len(old),
        "label_changes": label_changes,
        **{
            key: transitions[key]
            for key in (
                "wrong_to_correct",
                "correct_to_wrong",
                "correct_to_correct",
                "wrong_to_wrong",
            )
        },
        "accuracy_delta": (transitions["wrong_to_correct"] - transitions["correct_to_wrong"])
        / len(old),
    }


def compare_fairness(baseline_dir: Path, run_dirs: list[Path], output: Path) -> dict:
    baseline, baseline_records = scored_fairness(baseline_dir)
    baseline_lookup = {record["sample_id"]: record for record in baseline_records}
    base = fairness_summary(baseline, baseline_records)
    result = {
        "baseline_run": str(baseline_dir),
        "baseline_responses_sha256": sha256(baseline_dir / "responses.jsonl"),
        "baseline": base,
        "methods": {},
    }
    for run_dir in run_dirs:
        rows, records = scored_fairness(run_dir)
        manifest = json.loads((run_dir / "run.json").read_text())
        strategy = manifest["identity"]["strategy"]
        if strategy in result["methods"]:
            raise ValueError("Duplicate strategy in comparison.")
        summary = fairness_summary(rows, records)
        changes = paired_changes(baseline, rows)
        if (
            abs(
                summary["metrics"]["accuracy"]
                - base["metrics"]["accuracy"]
                - changes["accuracy_delta"]
            )
            > 1e-12
        ):
            raise AssertionError("Paired counts do not reproduce accuracy change.")
        body_changes = sum(
            record["response"]["choices"][0]["message"]["content"]
            != baseline_lookup[record["sample_id"]]["response"]["choices"][0]["message"]["content"]
            for record in records
        )
        total_cost = dict(summary["cost"])
        if strategy == "self_review":
            for key in ("requests", "prompt_tokens", "completion_tokens", "total_seconds"):
                total_cost[key] += base["cost"][key]
            total_cost["mean_seconds"] = total_cost["total_seconds"] / len(rows)
            total_cost["median_seconds"] = median(
                record["elapsed_seconds"] + baseline_lookup[record["sample_id"]]["elapsed_seconds"]
                for record in records
            )
        result["methods"][strategy] = {
            **summary,
            "paired": {**changes, "response_body_changes": body_changes},
            "run": str(run_dir),
            "responses_sha256": sha256(run_dir / "responses.jsonl"),
            "total_method_cost": total_cost,
        }
    write_json(output, result)
    return result
