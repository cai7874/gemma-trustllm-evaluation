"""Prepare and score a commonsense-only refinement after auditing justice labels."""

import argparse
import json
import random
import tomllib
from pathlib import Path

from a2.data import classification_sample, load_rows
from a2.experiment import validate_existing
from a2.io import read_jsonl, sha256, write_json, write_jsonl
from a2.tasks import CLASS_LABELS

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data/prepared/ethics_commonsense"
RUN = ROOT / "runs/ethics_commonsense"


def prepare() -> None:
    if PREPARED.exists():
        raise ValueError("The refined sample is already frozen; use its existing manifest.")
    config = tomllib.loads((ROOT / "config/part_a.toml").read_text())
    original = json.loads((ROOT / "data/prepared/part_a/manifest.json").read_text())
    excluded = {row["id"] for row in read_jsonl(ROOT / "data/prepared/part_a/development.jsonl")}
    raw_dir = ROOT / config["data"]["raw_dir"]
    for filename, expected_hash in original["source"]["files"].items():
        if sha256(raw_dir / filename) != expected_hash:
            raise ValueError(f"Source dataset changed: {filename}")
    raw = load_rows(raw_dir, "ethics")
    rng = random.Random(config["data"]["seed"])
    samples = []
    for label in sorted(CLASS_LABELS["implicit_ETHICS"]):
        candidates = [
            classification_sample("ethics", index, row)
            for index, row in enumerate(raw)
            if row["topic"] == "commonsense"
            and row["label"] == label
            and f"ethics:{index}" not in excluded
        ]
        rng.shuffle(candidates)
        if len(candidates) < 60:
            raise ValueError(f"Insufficient commonsense cases: {label}")
        samples.extend(candidates[:60])
    write_jsonl(PREPARED / "samples.jsonl", samples)
    write_json(
        PREPARED / "manifest.json",
        {
            "profile": "full",
            "purpose": "Part A commonsense-only ETHICS refinement",
            "source": original["source"],
            "seed": config["data"]["seed"],
            "sampling": {"topic": "commonsense", "per_label": 60, "excluded_ids": sorted(excluded)},
            "counts": {"ethics": 120},
            "input_count": 120,
            "samples_sha256": sha256(PREPARED / "samples.jsonl"),
            "rationale": "Post-run audit found justice task/label semantics incompatible "
            "with the shared wrong/not wrong prompt. No labels or prompts were repaired.",
        },
    )
    print("Frozen 120 commonsense examples; all original development IDs excluded.")


def score() -> None:
    from a2.analysis import classification_metrics, score_response

    samples = json.loads((RUN / "samples.json").read_text())
    records = read_jsonl(RUN / "responses.jsonl")
    if validate_existing(records, samples) != {row["id"] for row in samples}:
        raise ValueError("The commonsense run is incomplete.")
    by_id = {record["sample_id"]: record for record in records}
    rows = [score_response(sample, by_id[sample["id"]]) for sample in samples]
    result = {
        "inputs": len(rows),
        "responses_sha256": sha256(RUN / "responses.jsonl"),
        "metrics": classification_metrics(rows, CLASS_LABELS["implicit_ETHICS"]),
        "responses_with_reasoning_content": sum(row["has_reasoning_content"] for row in rows),
    }
    write_json(RUN / "scores.json", result)
    write_json(RUN / "scored_samples.json", rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        score()
