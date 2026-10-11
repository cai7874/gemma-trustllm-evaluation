"""Match TrustLLM justice text to the original author's pinned ETHICS labels."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from a2.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
REPO = "hendrycks/ethics"
REVISION = "b8b47c589f8bee77175b8648e5497278b68da48a"
CACHE = ROOT / ".cache/ethics_original/hf"


def main(download: bool) -> None:
    lookup, hashes = defaultdict(set), {}
    for split in ("train", "test", "test_hard"):
        filename = f"data/justice/{split}.csv"
        if download:
            hf_hub_download(
                REPO,
                filename,
                repo_type="dataset",
                revision=REVISION,
                local_dir=CACHE,
                cache_dir=ROOT / ".cache/huggingface",
            )
        path = CACHE / filename
        hashes[filename] = sha256(path)
        with path.open(newline="") as file:
            for row in csv.reader(file):
                if row[0] in {"0", "1"}:
                    lookup[row[1]].add(int(row[0]))
    rows = json.loads((ROOT / "data/raw/trustllm/ethics/implicit_ETHICS.json").read_text())
    counts, unmatched = Counter(), []
    for index, row in enumerate(rows):
        if row["topic"] != "justice":
            continue
        labels = lookup.get(row["input"])
        if labels:
            counts[(row["label"], tuple(sorted(labels)))] += 1
        else:
            unmatched.append(index)
    result = {
        "source_repo": REPO,
        "source_revision": REVISION,
        "files_sha256": hashes,
        "justice_rows": 250,
        "match_count": sum(counts.values()),
        "mapping_counts": [
            {"trustllm_label": label, "original_numeric_labels": list(numbers), "count": count}
            for (label, numbers), count in counts.items()
        ],
        "unmatched_source_rows": unmatched,
        "interpretation": "Original justice predicts reasonable/unreasonable claims, "
        "not commonsense wrong/not wrong actions. Source-paper task "
        "semantics and numeric polarity must be considered together.",
    }
    write_json(ROOT / "runs/part_a_baseline/ethics_label_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    main(parser.parse_args().download)
