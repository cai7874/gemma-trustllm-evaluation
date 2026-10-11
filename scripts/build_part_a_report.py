"""Export audited LaTeX tables and manually selected cases from the frozen run."""

import json
from collections import Counter
from pathlib import Path

from a2.io import read_jsonl, sha256, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/part_a_baseline"
ETHICS_RUN = ROOT / "runs/ethics_commonsense"
REPORT = ROOT / "report"


def tex(text: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(character, character) for character in text)


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}\\%"


def table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    columns = "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{table}[H]\centering\small", rf"\begin{{tabular}}{{{columns}}}"]
    lines += [r"\toprule", " & ".join(headers) + r"\\", r"\midrule"]
    lines += [" & ".join(row) + r"\\" for row in rows]
    lines += [r"\bottomrule\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def export_tables(summary: dict, rows: list[dict]) -> None:
    classification = []
    for dimension in ("fairness", "ethics"):
        result = summary[dimension]
        classification.append(
            [
                dimension.title(),
                str(result["inputs"]),
                percent(result["accuracy"]),
                f"{result['macro_f1']:.3f}",
                percent(result["balanced_accuracy"]),
            ]
        )
    text = table(
        ["Subset", "$n$", "Accuracy", "Macro-F1", "Balanced accuracy"],
        classification,
        "Classification results. Equal reference-class counts make accuracy and "
        "balanced accuracy equal in these two selected samples.",
    )
    (REPORT / "classification_table.tex").write_text(text)
    robust = summary["robustness"]
    task_results = {**robust["by_task"], "pooled": robust["overall"]}
    text = table(
        ["Task", "Pairs", "Original", "Modified", "$\\Delta$ (pp)", "ASR", "$a/o$"],
        [
            [
                name.upper(),
                str(result["pairs"]),
                percent(result["original_accuracy"]),
                percent(result["modified_accuracy"]),
                f"{100 * result['accuracy_drop']:.1f}",
                percent(result["attack_success_rate"]),
                f"{result['correct_to_wrong']}/{result['asr_denominator']}",
            ]
            for name, result in task_results.items()
        ],
        "Paired robustness results. Positive $\\Delta$ means lower perturbed accuracy; "
        "$a/o$ gives successful attacks over originally correct cases.",
    )
    (REPORT / "robustness_table.tex").write_text(text)
    recall_rows = []
    for dimension in ("fairness", "ethics"):
        subset = [row for row in rows if row["dimension"] == dimension]
        for label, recall in summary[dimension]["recall_by_label"].items():
            matching = [row for row in subset if row["gold_label"] == label]
            recall_rows.append(
                [
                    f"{dimension.title()}: {tex(label)}",
                    str(len(matching)),
                    str(sum(row["correct"] for row in matching)),
                    percent(recall),
                ]
            )
    (REPORT / "recall_table.tex").write_text(
        table(
            ["Reference class", "$n$", "Correct", "Recall"],
            recall_rows,
            "Class-specific recall includes abstentions and invalid output as errors.",
        )
    )
    topic_rows = []
    for topic in ("race", "profession", "gender", "religion"):
        subset = [
            row
            for row in rows
            if row["dimension"] == "fairness" and row["metadata"]["type"] == topic
        ]
        topic_rows.append(
            [
                topic.title(),
                str(len(subset)),
                percent(sum(row["correct"] for row in subset) / len(subset)),
            ]
        )
    (REPORT / "topic_table.tex").write_text(
        table(
            ["Fairness topic", "$n$", "Accuracy"],
            topic_rows,
            "Exploratory topic results; unequal class mixtures and small strata prevent "
            "interpreting these values as demographic fairness guarantees.",
        )
    )


def export_cases(rows: list[dict], records: dict[str, dict]) -> dict:
    selected = json.loads((REPORT / "failure_selection.json").read_text())
    by_id = {row["id"]: row for row in rows}
    pairs = {}
    for row in rows:
        if row["dimension"] == "robustness":
            pairs.setdefault(row["pair_id"], {})[row["variant"]] = row
    case_dir = REPORT / "cases"
    case_dir.mkdir(exist_ok=True)
    lines, audited = [], {}
    for dimension, prefix in (("fairness", "F"), ("ethics", "E"), ("robustness", "R")):
        choices = selected[dimension]
        if len(choices) < 5 or len({case["id"] for case in choices}) != len(choices):
            raise ValueError(f"Need at least five distinct selected cases: {dimension}")
        lines.append(rf"\subsection{{{dimension.title()}}}")
        audited[dimension] = []
        for number, case in enumerate(choices, start=1):
            case_id = f"{prefix}{number}"
            examples = (
                [
                    ("original", pairs[case["id"]]["original"]),
                    ("modified", pairs[case["id"]]["modified"]),
                ]
                if dimension == "robustness"
                else [("input", by_id[case["id"]])]
            )
            failing = examples[-1][1]
            if failing["correct"] or failing["dimension"] != dimension:
                raise ValueError(f"Selected case is not a scored error: {case['id']}")
            if dimension == "robustness" and not examples[0][1]["correct"]:
                raise ValueError(f"Selected attack requires a correct original: {case['id']}")
            lines.append(r"\Needspace{20\baselineskip}")
            lines.append(rf"\subsubsection*{{{case_id}: {tex(case['title'])}}}")
            lines.append(
                rf"\textbf{{Reference label:}} {tex(failing['gold_label'])}. "
                rf"\textbf{{Case ID:}} \texttt{{{tex(case['id'])}}}."
            )
            for variant, row in examples:
                stem = f"{case_id}_{variant}"
                response = records[row["id"]]["response"]["choices"][0]["message"]["content"]
                (case_dir / f"{stem}_prompt.txt").write_text(row["prompt"])
                (case_dir / f"{stem}_response.txt").write_text(response)
                lines.append(r"\Needspace{14\baselineskip}")
                lines.append(r"\begin{samepage}")
                lines.append(
                    rf"\textbf{{{variant.title()} prompt and response}} "
                    rf"(prediction: {tex(row['prediction'])})."
                )
                for kind in ("prompt", "response"):
                    lines.append(
                        rf"\lstinputlisting[style=response]"
                        rf"{{../report/cases/{stem}_{kind}.txt}}"
                    )
                lines.append(r"\end{samepage}")
            lines.append(r"\textbf{Interpretation:} " + tex(case["analysis"]))
            audited[dimension].append({**case, "examples": [row for _, row in examples]})
    (REPORT / "failure_examples.tex").write_text("\n\n".join(lines) + "\n")
    write_json(RUN / "selected_failure_cases.json", audited)
    return {dimension: len(cases) for dimension, cases in audited.items()}


def export_code() -> None:
    files = [
        *sorted((ROOT / "src/a2").glob("*.py")),
        ROOT / "tests/test_evaluation.py",
        *sorted((ROOT / "scripts").glob("*.py")),
        ROOT / "config/part_a.toml",
        ROOT / "pyproject.toml",
    ]
    lines = [
        "From the project root, rebuild the environment with "
        r"\texttt{uv sync --locked}; start the local LM Studio server; then run:",
        r"\begin{lstlisting}[style=response]",
        "uv run a2 check",
        "uv run a2 prepare --profile full --output data/prepared/part_a",
        "uv run a2 generate --prepared data/prepared/part_a --run runs/part_a_baseline",
        "uv run a2 score --run runs/part_a_baseline",
        "uv run a2 report --run runs/part_a_baseline",
        "uv run python scripts/audit_ethics_labels.py --download",
        "uv run python scripts/ethics_commonsense.py prepare",
        "uv run a2 generate --prepared data/prepared/ethics_commonsense "
        "--run runs/ethics_commonsense",
        "uv run python scripts/ethics_commonsense.py score",
        "uv run python scripts/build_part_a_report.py",
        r"\end{lstlisting}",
        "Preparation and generation refuse to overwrite existing experiments. "
        "Use fresh directories for a new run. Offline scoring/reporting can be repeated.",
    ]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        lines.append(rf"\subsection*{{\nolinkurl{{{relative}}}}}")
        language = ",language=Python" if path.suffix == ".py" else ""
        lines.append(rf"\lstinputlisting[style=code{language}]{{../{relative}}}")
    (REPORT / "code_appendix.tex").write_text("\n\n".join(lines) + "\n")


def main() -> None:
    import pandas as pd

    from a2.analysis import save_confusion_plot

    summary = json.loads((RUN / "scores.json").read_text())
    if summary["profile"] != "full" or summary["input_count"] != 480:
        raise ValueError("This report requires the frozen 480-input Part A run.")
    if sha256(RUN / "responses.jsonl") != summary["responses_sha256"]:
        raise ValueError("Responses changed after scoring.")
    original_rows = json.loads((RUN / "scored_samples.json").read_text())
    original_records = read_jsonl(RUN / "responses.jsonl")
    ethics_summary = json.loads((ETHICS_RUN / "scores.json").read_text())
    if sha256(ETHICS_RUN / "responses.jsonl") != ethics_summary["responses_sha256"]:
        raise ValueError("Commonsense responses changed after scoring.")
    ethics_rows = json.loads((ETHICS_RUN / "scored_samples.json").read_text())
    if len(ethics_rows) != 120 or any(
        row["metadata"]["topic"] != "commonsense" for row in ethics_rows
    ):
        raise ValueError("The primary ethics result requires 120 commonsense cases.")
    baseline_ids = {row["id"] for row in original_rows if row["dimension"] != "ethics"}
    rows = [row for row in original_rows if row["id"] in baseline_ids] + ethics_rows
    record_list = [record for record in original_records if record["sample_id"] in baseline_ids]
    record_list += read_jsonl(ETHICS_RUN / "responses.jsonl")
    summary["original_mixed_ethics"] = summary["ethics"]
    summary["ethics"] = ethics_summary["metrics"]
    summary["source_runs"] = {
        "fairness_and_robustness": "runs/part_a_baseline",
        "ethics": "runs/ethics_commonsense",
    }
    summary["ethics_responses_sha256"] = ethics_summary["responses_sha256"]
    summary["answer_status_counts"] = dict(Counter(row["answer_status"] for row in rows))
    summary["responses_with_reasoning_content"] = sum(row["has_reasoning_content"] for row in rows)
    records = {record["sample_id"]: record for record in record_list}
    if len(records) != 480 or len(record_list) != 480:
        raise ValueError("Missing or duplicate responses.")
    samples_by_id = {
        sample["id"]: sample
        for run in (RUN, ETHICS_RUN)
        for sample in json.loads((run / "samples.json").read_text())
    }
    primary_samples = [samples_by_id[row["id"]] for row in rows]
    write_jsonl(REPORT / "primary_samples.jsonl", primary_samples)
    write_jsonl(REPORT / "primary_responses.jsonl", record_list)
    summary["baseline_responses_sha256"] = summary["responses_sha256"]
    summary["responses_sha256"] = sha256(REPORT / "primary_responses.jsonl")
    summary["samples_sha256"] = sha256(REPORT / "primary_samples.jsonl")
    summary["purpose"] = "Part A primary results after documented commonsense refinement"
    export_tables(summary, rows)
    counts = export_cases(rows, records)
    export_code()
    audit = {
        "inputs": len(rows),
        "selected_failure_counts": counts,
        "answer_status_counts": dict(Counter(row["answer_status"] for row in rows)),
        "primary_inference_seconds": sum(record["elapsed_seconds"] for record in record_list),
        "response_tokens": sum(
            record["response"]["usage"]["completion_tokens"] for record in record_list
        ),
        "requests_with_retry": sum(len(record["attempts"]) > 1 for record in record_list),
        "responses_sha256": summary["responses_sha256"],
        "ethics_responses_sha256": summary["ethics_responses_sha256"],
        "total_generated_inputs_including_original_mixed_ethics": 600,
    }
    write_json(RUN / "report_audit.json", audit)
    write_json(REPORT / "primary_scores.json", summary)
    write_json(REPORT / "primary_scored_samples.json", rows)
    pd.DataFrame(rows).drop(columns=["source_record"]).to_csv(
        REPORT / "primary_scored_samples.csv", index=False
    )
    (ETHICS_RUN / "figures").mkdir(exist_ok=True)
    save_confusion_plot(
        summary["ethics"],
        "Part A - Commonsense moral judgment",
        ETHICS_RUN / "figures/ethics_confusion.png",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
