"""Persist every response and reject incompatible attempts to resume a run."""

import json
import platform
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from tqdm import tqdm

from a2.client import LMStudioClient
from a2.io import read_jsonl, sha256, write_json
from a2.tasks import PROMPT_VERSION


def validate_existing(records: list[dict], samples: list[dict]) -> set[str]:
    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate responses found; the run must be inspected.")
    if not set(ids) <= {sample["id"] for sample in samples}:
        raise ValueError("Responses include samples outside the prepared input list.")
    return set(ids)


def generate(config: dict, prepared_dir: Path, run_dir: Path, resume: bool) -> dict:
    samples = read_jsonl(prepared_dir / "samples.jsonl")
    if not samples:
        raise ValueError("No prepared samples found.")
    prepared_manifest = json.loads((prepared_dir / "manifest.json").read_text())
    sample_hash = sha256(prepared_dir / "samples.jsonl")
    if prepared_manifest["samples_sha256"] != sample_hash:
        raise ValueError("Prepared samples changed after preparation.")
    identity = {
        "config": config,
        "samples_sha256": sample_hash,
        "prompt_version": PROMPT_VERSION,
        "code_sha256": {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    manifest_path = run_dir / "run.json"
    if run_dir.exists():
        if not resume or not manifest_path.exists():
            raise ValueError("Use a new run directory or explicitly resume an existing run.")
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Configuration, samples, prompts, or code changed; create a new run.")
    else:
        manifest = {
            "identity": identity,
            "prepared_manifest": prepared_manifest,
            "started_at": datetime.now(UTC).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    name: version(name)
                    for name in (
                        "openai",
                        "huggingface-hub",
                        "pandas",
                        "scikit-learn",
                        "matplotlib",
                    )
                },
            },
        }
    client = LMStudioClient(config["model"])
    connection = client.connection_info()
    if (
        "connection" in manifest
        and connection["target_model"] != manifest["connection"]["target_model"]
    ):
        raise ValueError("Model metadata changed since this run started.")
    manifest["connection"] = connection
    write_json(manifest_path, manifest)
    write_json(run_dir / "samples.json", samples)
    response_path = run_dir / "responses.jsonl"
    existing = read_jsonl(response_path)
    seen = validate_existing(existing, samples)
    if any(record["status"] != "completed" for record in existing):
        raise ValueError(
            "The run contains a technical failure; inspect it and use a new run directory."
        )
    pending = [sample for sample in samples if sample["id"] not in seen]
    for sample in tqdm(pending, desc="Generating", unit="input"):
        result = client.generate(sample)
        # Closing the append operation after each input makes completed work durable.
        with response_path.open("a") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        if result["status"] == "request_failed":
            raise RuntimeError(f"Request failed for {sample['id']}; inspect responses.jsonl.")
    records = read_jsonl(response_path)
    summary = {
        "inputs": len(samples),
        "recorded": len(records),
        "status_counts": dict(Counter(row["status"] for row in records)),
    }
    write_json(run_dir / "generation_summary.json", summary)
    return summary
