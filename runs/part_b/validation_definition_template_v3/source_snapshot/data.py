"""Download versioned data and build disjoint development/test samples."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from a2.io import sha256, write_json, write_jsonl
from a2.tasks import CLASS_LABELS, ROBUSTNESS_LABELS

DATA_FILES = {
    "fairness": "fairness/stereotype_recognition.json",
    "robustness": "robustness/AdvGLUE.json",
    "ethics": "ethics/implicit_ETHICS.json",
}


def download_data(config: dict, project_root: Path) -> dict:
    settings = config["data"]
    raw_dir = project_root / settings["raw_dir"]
    for filename in DATA_FILES.values():
        hf_hub_download(
            repo_id=settings["repo_id"],
            repo_type="dataset",
            filename=filename,
            revision=settings["revision"],
            local_dir=raw_dir,
            cache_dir=project_root / ".cache/huggingface",
        )
    source = {
        "provider": "huggingface",
        "repo_id": settings["repo_id"],
        "revision": settings["revision"],
        "files": {name: sha256(raw_dir / name) for name in DATA_FILES.values()},
    }
    write_json(raw_dir / "source.json", source)
    return source


def load_rows(raw_dir: Path, dimension: str) -> list[dict]:
    rows = json.loads((raw_dir / DATA_FILES[dimension]).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON list for {dimension}.")
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Missing prompt in {dimension}.")
    return rows


def classification_sample(dimension: str, index: int, row: dict) -> dict:
    task = "stereotype_recognition" if dimension == "fairness" else "implicit_ETHICS"
    if row["label"] not in CLASS_LABELS[task]:
        raise ValueError(f"Unexpected {task} label: {row['label']!r}")
    return {
        "id": f"{dimension}:{index}",
        "dimension": dimension,
        "task": task,
        "prompt": row["prompt"],
        "gold_label": row["label"],
        "source_row": index,
        "metadata": {key: row[key] for key in ("type", "topic") if key in row},
        "source_record": row,
    }


def split_classification(
    rows: list[dict], dimension: str, settings: dict, seed: int
) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        sample = classification_sample(dimension, index, row)
        groups[sample["gold_label"]].append(sample)
    rng = random.Random(seed)
    development, test = [], []
    total = settings[f"{dimension}_test"]
    labels = sorted(groups)
    for position, label in enumerate(labels):
        candidates = groups[label]
        rng.shuffle(candidates)
        dev_count = settings["development_per_label"]
        test_count = total // len(labels) + (position < total % len(labels))
        if len(candidates) < dev_count + test_count:
            raise ValueError(f"Not enough samples for {dimension}/{label}.")
        development.extend(candidates[:dev_count])
        test.extend(candidates[dev_count : dev_count + test_count])
    return development, test


def robustness_pairs(rows: list[dict]) -> dict[str, list[list[dict]]]:
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        task = row["task"]
        if task not in ROBUSTNESS_LABELS or row["type"] not in {"original", "modified"}:
            raise ValueError(f"Unknown robustness task or variant at row {index}.")
        pair_id = f"{task}:{row['index']}"
        grouped[pair_id].append(
            {
                "id": f"robustness:{pair_id}:{row['type']}",
                "dimension": "robustness",
                "task": task,
                "prompt": row["prompt"],
                "gold_label": ROBUSTNESS_LABELS[task][row["label"]],
                "pair_id": pair_id,
                "variant": row["type"],
                "source_row": index,
                "metadata": {key: row.get(key) for key in ("method", "data construction")},
                "source_record": row,
            }
        )
    by_task = defaultdict(list)
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {item["variant"] for item in pair} != {"original", "modified"}:
            raise ValueError(f"Incomplete or duplicated pair: {pair_id}")
        if pair[0]["gold_label"] != pair[1]["gold_label"]:
            raise ValueError(f"Conflicting gold labels: {pair_id}")
        by_task[pair[0]["task"]].append(sorted(pair, key=lambda item: item["variant"]))
    if set(by_task) != set(ROBUSTNESS_LABELS):
        raise ValueError("The dataset does not contain all four AdvGLUE tasks.")
    return by_task


def split_robustness(rows: list[dict], settings: dict, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    development, test = [], []
    for task, pairs in sorted(robustness_pairs(rows).items()):
        rng.shuffle(pairs)
        dev_count = settings["development_pairs_per_task"]
        test_count = settings["robustness_test_pairs_per_task"]
        if len(pairs) < dev_count + test_count:
            raise ValueError(f"Not enough robustness pairs for {task}.")
        development.extend(item for pair in pairs[:dev_count] for item in pair)
        test.extend(item for pair in pairs[dev_count : dev_count + test_count] for item in pair)
    return development, test


def smoke_samples(development: list[dict]) -> list[dict]:
    """Use one case per classification label and one complete pair per GLUE task."""
    selected, seen = [], set()
    for sample in development:
        key = (sample["dimension"], sample["task"], sample.get("gold_label"))
        if sample["dimension"] == "robustness":
            key = ("robustness", sample["task"])
            if key not in seen:
                pair_id = sample["pair_id"]
                selected.extend(item for item in development if item.get("pair_id") == pair_id)
                seen.add(key)
        elif key not in seen:
            selected.append(sample)
            seen.add(key)
    return selected


def prepare(config: dict, project_root: Path, output_dir: Path, profile: str) -> dict:
    if output_dir.exists():
        raise ValueError(f"Prepared directory already exists: {output_dir}")
    raw_dir = project_root / config["data"]["raw_dir"]
    source = json.loads((raw_dir / "source.json").read_text())
    if source["provider"] != "huggingface" or source["revision"] != config["data"]["revision"]:
        raise ValueError("Dataset provenance does not match the configured Hugging Face revision.")
    for filename, expected_hash in source["files"].items():
        if sha256(raw_dir / filename) != expected_hash:
            raise ValueError(f"Dataset file changed: {filename}")
    development, test, overview = [], [], {}
    for dimension in DATA_FILES:
        rows = load_rows(raw_dir, dimension)
        settings, seed = config["sampling"], config["data"]["seed"]
        if dimension == "robustness":
            dev_part, test_part = split_robustness(rows, settings, seed)
        else:
            dev_part, test_part = split_classification(rows, dimension, settings, seed)
        development.extend(dev_part)
        test.extend(test_part)
        overview[dimension] = {
            "rows": len(rows),
            "fields": sorted(rows[0]),
            "labels": dict(Counter(str(row["label"]) for row in rows)),
            "groups": dict(
                Counter(str(row.get("task", row.get("type", row.get("topic")))) for row in rows)
            ),
        }
    if {item["id"] for item in development} & {item["id"] for item in test}:
        raise ValueError("Development and test samples overlap.")
    samples = smoke_samples(development) if profile == "smoke" else test
    write_jsonl(output_dir / "samples.jsonl", samples)
    write_jsonl(output_dir / "development.jsonl", development)
    manifest = {
        "profile": profile,
        "purpose": "development smoke test" if profile == "smoke" else "Part A test set",
        "source": source,
        "sampling": config["sampling"],
        "seed": config["data"]["seed"],
        "overview": overview,
        "counts": dict(Counter(item["dimension"] for item in samples)),
        "input_count": len(samples),
        "samples_sha256": sha256(output_dir / "samples.jsonl"),
        "formal_test_ids_reserved": [item["id"] for item in test],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
