"""Explicit commands for preparation, generation, scoring, and report export."""

import argparse
import json
import logging
import os
import tomllib
from pathlib import Path

from a2.client import LMStudioClient
from a2.data import download_data, prepare
from a2.experiment import generate
from a2.io import write_json


def load_config(path: Path) -> tuple[dict, Path]:
    path = path.resolve()
    config = tomllib.loads(path.read_text())
    if config["model"]["max_attempts"] < 1 or config["model"]["max_tokens"] < 1:
        raise ValueError("Request attempts and token limit must be positive.")
    return config, path.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustLLM evaluation through local LM Studio.")
    parser.add_argument("--config", type=Path, default=Path("config/part_a.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check the served model and download HF data.")
    check.add_argument("--output", type=Path, default=Path("runs/connection_check/checks.json"))
    preparation = commands.add_parser(
        "prepare", help="Prepare smoke or formal samples; no inference."
    )
    preparation.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    preparation.add_argument("--output", type=Path, required=True)
    generation = commands.add_parser(
        "generate", help="Generate answers for a prepared sample list."
    )
    generation.add_argument("--prepared", type=Path, required=True)
    generation.add_argument("--run", type=Path, required=True)
    generation.add_argument("--resume", action="store_true")
    for name, help_text in [
        ("score", "Score saved responses offline."),
        ("report", "Export tables, figures, and report notes offline."),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    if args.command == "check":
        result = {
            "lm_studio": LMStudioClient(config["model"]).connection_info(),
            "huggingface": download_data(config, root),
        }
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "prepare":
        result = prepare(config, root, args.output, args.profile)
        print(
            json.dumps(
                {key: result[key] for key in ("profile", "purpose", "counts", "input_count")},
                indent=2,
            )
        )
    elif args.command == "generate":
        print(json.dumps(generate(config, args.prepared, args.run, args.resume), indent=2))
    elif args.command == "score":
        from a2.analysis import score

        print(json.dumps(score(args.run), indent=2))
    elif args.command == "report":
        from a2.analysis import export_report

        print(f"Report materials exported to {export_report(args.run)}")
