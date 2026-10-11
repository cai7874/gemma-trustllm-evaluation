"""Build the combined report from frozen Part A and completed Part B records."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_part_a_report import percent, table, tex

from a2.analysis import save_confusion_plot
from a2.comparison import compare_fairness, scored_fairness
from a2.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/part_b"
NAMES = {
    "baseline": "Part A baseline",
    "definition_template": "Definition rubric",
    "few_shot": "Six-shot alignment",
}
RUNS = {name: ROOT / "runs/part_b" / name for name in ("definition_template", "few_shot")}


def export_tables(result: dict) -> None:
    conditions = {"baseline": result["baseline"], **result["methods"]}
    comparison_rows, recall_rows, topic_rows, cost_rows, change_rows = [], [], [], [], []
    for name, condition in conditions.items():
        metrics = condition["metrics"]
        delta = condition.get("paired", {}).get("accuracy_delta", 0)
        comparison_rows.append(
            [
                NAMES[name],
                percent(metrics["accuracy"]),
                f"{metrics['macro_f1']:.3f}",
                f"{100 * delta:+.1f}",
            ]
        )
        recall_rows.append(
            [NAMES[name], *[percent(v) for v in metrics["recall_by_label"].values()]]
        )
        topic_rows.append(
            [
                NAMES[name],
                *[
                    percent(condition["by_topic"][t]["accuracy"])
                    for t in ("race", "profession", "gender", "religion")
                ],
            ]
        )
        cost = condition["cost"]
        cost_rows.append(
            [
                NAMES[name],
                str(cost["requests"]),
                str(cost["prompt_tokens"]),
                str(cost["completion_tokens"]),
                f"{cost['mean_seconds']:.2f}",
                f"{cost['median_seconds']:.2f}",
            ]
        )
        if name != "baseline":
            changes = condition["paired"]
            change_rows.append(
                [
                    NAMES[name],
                    str(changes["label_changes"]),
                    str(changes["wrong_to_correct"]),
                    str(changes["correct_to_wrong"]),
                    str(changes["wrong_to_wrong"]),
                    str(changes["response_body_changes"]),
                ]
            )
    exports = {
        "comparison_table.tex": (
            ["Condition", "Accuracy", "Macro-F1", "$\\Delta$ (pp)"],
            comparison_rows,
            "All conditions use the same 120 test inputs; delta is relative to Part A. "
            "Balanced accuracy equals accuracy because each class has 40 items.",
        ),
        "recall_table.tex": (
            ["Condition", "Stereotype", "Anti-stereotype", "Unrelated"],
            recall_rows,
            "Class recalls expose gains and losses hidden by overall accuracy.",
        ),
        "changes_table.tex": (
            ["Method", "Changed", "Fixed", "Damaged", "Still wrong", "New text"],
            change_rows,
            "Paired prediction changes. Fixed means wrong-to-correct; "
            "damaged means correct-to-wrong. New text counts any raw JSON content change, "
            "including explanation-only changes.",
        ),
        "topic_table.tex": (
            ["Condition", "Race (50)", "Profession (46)", "Gender (17)", "Religion (7)"],
            topic_rows,
            "Descriptive topic accuracy with sample counts; "
            "these domains are not demographic parity groups.",
        ),
        "cost_table.tex": (
            ["Condition", "Calls", "Input tokens", "Output tokens", "Mean (s)", "Median (s)"],
            cost_rows,
            "Measured request latency and token cost; both selected mitigations use "
            "one call per input. Model mistakes are not retried.",
        ),
    }
    for filename, (headers, rows, caption) in exports.items():
        (REPORT / filename).write_text(table(headers, rows, caption))
    pd.DataFrame(
        [
            {
                "condition": name,
                "accuracy": c["metrics"]["accuracy"],
                "macro_f1": c["metrics"]["macro_f1"],
                **c["metrics"]["recall_by_label"],
            }
            for name, c in conditions.items()
        ]
    ).to_csv(REPORT / "metrics.csv", index=False)


def export_figures(result: dict) -> None:
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    conditions = {"baseline": result["baseline"], **result["methods"]}
    colors = ["#366b9c", "#d18445", "#4b8768"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    names = list(conditions)
    values = [c["metrics"]["accuracy"] for c in conditions.values()]
    bars = axes[0].bar(range(3), values, color=colors)
    axes[0].set_xticks(range(3), ["Baseline", "Rubric", "Six-shot"])
    axes[0].set(title="Accuracy on the same 120 inputs", ylabel="Accuracy", ylim=(0, 1.08))
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center")
    x = np.arange(3)
    for index, name in enumerate(names):
        axes[1].bar(
            x + (index - 1) * 0.24,
            list(conditions[name]["metrics"]["recall_by_label"].values()),
            width=0.24,
            label=NAMES[name],
            color=colors[index],
        )
    axes[1].set_xticks(x, ["Stereotype", "Anti-stereotype", "Unrelated"])
    axes[1].set(title="Recall by reference class", ylabel="Recall", ylim=(0, 1.3))
    axes[1].legend(fontsize=8, loc="upper left")
    figure.savefig(figures / "comparison.png", dpi=180)
    plt.close(figure)
    for name, condition in conditions.items():
        save_confusion_plot(condition["metrics"], NAMES[name], figures / f"{name}_confusion.png")


def export_cases() -> None:
    selections = json.loads((REPORT / "case_selection.json").read_text())
    baseline, base_records = scored_fairness(ROOT / "runs/part_a_baseline")
    before = {row["id"]: row for row in baseline}
    records = {"baseline": {r["sample_id"]: r for r in base_records}}
    rows = {}
    for name, run in RUNS.items():
        scored, raw = scored_fairness(run)
        rows[name] = {r["id"]: r for r in scored}
        records[name] = {r["sample_id"]: r for r in raw}
    destination = REPORT / "cases"
    destination.mkdir(exist_ok=True)
    lines, audit = [], {}
    for name, cases in selections.items():
        if len(cases) < 5 or len({c["id"] for c in cases}) != len(cases):
            raise ValueError("Need five distinct inspected cases per method.")
        lines.append(rf"\subsection{{{NAMES[name]}}}")
        audit[name] = []
        for index, case in enumerate(cases, 1):
            sample_id = case["id"]
            old, new = before[sample_id], rows[name][sample_id]
            kind = (
                "correction"
                if not old["correct"] and new["correct"]
                else "regression"
                if old["correct"] and not new["correct"]
                else "remaining error"
                if not new["correct"]
                else "preserved correct"
            )
            if kind != case["kind"]:
                raise ValueError(
                    f"Case classification differs from its saved outcomes: {sample_id}"
                )
            stem = f"{name}_{index}"
            (destination / f"{stem}_prompt.txt").write_text(old["prompt"])
            for label in ("baseline", name):
                content = records[label][sample_id]["response"]["choices"][0]["message"]["content"]
                (destination / f"{stem}_{label}.txt").write_text(content)
            lines += [
                r"\Needspace{23\baselineskip}",
                rf"\subsubsection*{{{index}. {tex(case['title'])}}}",
                rf"\textbf{{Case:}} \texttt{{{sample_id}}}. "
                rf"\textbf{{Reference:}} {tex(old['gold_label'])}. "
                rf"\textbf{{Outcome:}} {kind}.",
                r"\begin{samepage}",
                rf"\lstinputlisting[style=response]{{../report/part_b/cases/{stem}_prompt.txt}}",
                rf"\textbf{{Baseline:}} {tex(old['prediction'])}.",
                rf"\lstinputlisting[style=response]{{../report/part_b/cases/{stem}_baseline.txt}}",
                rf"\textbf{{{NAMES[name]}:}} {tex(new['prediction'])}.",
                rf"\lstinputlisting[style=response]{{../report/part_b/cases/{stem}_{name}.txt}}",
                r"\end{samepage}",
                r"\textbf{Interpretation:} " + tex(case["analysis"]),
            ]
            audit[name].append({**case, "before": old, "after": new})
    (REPORT / "case_examples.tex").write_text("\n\n".join(lines) + "\n")
    write_json(REPORT / "selected_cases.json", audit)


def export_code() -> None:
    files = [
        *sorted((ROOT / "src/a2").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        ROOT / "config/part_a.toml",
        ROOT / "config/part_b.toml",
        ROOT / "pyproject.toml",
    ]
    lines = [
        "All implemented Python code and configuration are included below. "
        "The private repository is a supplementary backup, not a prerequisite for "
        "reading this submission. The Part A checkpoint is tagged ",
        r"\texttt{part-a}; the final Part B merge is tagged \texttt{part-b}.",
        "Part A's original client, CLI, and experiment runner are also included "
        "after the current implementation; the other original modules are unchanged.",
        r"\lstinputlisting[style=response]{../report/part_b/reproduction_commands.txt}",
    ]
    files += [
        ROOT / "runs/part_a_baseline/source_snapshot" / name
        for name in ("client.py", "cli.py", "experiment.py")
    ]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        language = ",language=Python" if path.suffix == ".py" else ""
        lines += [
            r"\Needspace{5\baselineskip}",
            rf"\subsection*{{\nolinkurl{{{relative}}}}}",
            rf"\lstinputlisting[style=code{language}]{{../{relative}}}",
        ]
    (REPORT / "code_appendix.tex").write_text("\n\n".join(lines) + "\n")


def export_combined_sources(result: dict) -> None:
    source = (ROOT / "report/introduction_setup.tex").read_text()
    source = source.replace(
        "This study evaluates one locally deployed open-weight",
        "Part A evaluates one locally deployed open-weight",
    )
    source = source.replace(
        "Each input was an independent", "In Part A, each input was an independent"
    )
    source = source.replace(
        "experiment generated 600 responses in total", "Part A generated 600 responses in total"
    )
    (REPORT / "introduction_setup.tex").write_text(source)
    references = (
        (ROOT / "report/references.tex")
        .read_text()
        .replace(
            r"\end{thebibliography}",
            r"""\bibitem{fewshot} T. Brown et al. \emph{Language Models are Few-Shot Learners}.
NeurIPS, 2020. \url{https://arxiv.org/abs/2005.14165}.
\bibitem{selfrefine} A. Madaan et al. \emph{Self-Refine: Iterative Refinement with Self-Feedback}.
NeurIPS, 2023. \url{https://arxiv.org/abs/2303.17651}.
\bibitem{selfcorrect} J. Huang et al. \emph{Large Language Models Cannot
Self-Correct Reasoning Yet}.
ICLR, 2024. \url{https://arxiv.org/abs/2310.01798}.
\end{thebibliography}""",
        )
    )
    (REPORT / "references.tex").write_text(references)
    discussion = (ROOT / "report/discussion.tex").read_text()
    discussion = discussion.replace(
        r"\subsection*{Use of generative AI}",
        r"""\subsection{Part B interpretation}
The mitigation comparison evaluates improvements in stereotype recognition under
the same reference labels. It does not demonstrate reduced bias in open-ended
generation or equitable decisions about people. The class and topic results show
where improvement transfers and where errors remain. Prompt-based interventions
can improve agreement with an annotation convention while amplifying ambiguous
associations; human review remains necessary for consequential uses.

The initial development attempts, later prompt revisions, and replacement of
self-review were recorded before formal evaluation. All final test outputs were
scored once without adjusting either method after observing its test accuracy.
This was an informed follow-up on previously inspected Part A data, not a blind
held-out study. A new context-disjoint test set and more model configurations would
be needed to assess generalization. The 120 fairness items share 110 contexts;
no independent-item significance claim is made.

\subsection*{Use of generative AI}""",
    )
    discussion = discussion.replace(
        "prepare this report draft.",
        "prepare this report draft, including Part B mitigation code, "
        "development revisions, and comparison analysis.",
    )
    (REPORT / "discussion.tex").write_text(discussion)
    master = (ROOT / "1RT730_Assignment2/assignment2.tex").read_text()
    master = master.replace("Part A TrustLLM Evaluation", "Part A and Part B TrustLLM Evaluation")
    master = master.replace(
        "Part A: Local Gemma 4 E4B Evaluation with TrustLLM",
        "Local Gemma 4 E4B: TrustLLM Evaluation and Fairness Mitigation",
    )
    master = master.replace(
        "../report/introduction_setup.tex", "../report/part_b/introduction_setup.tex"
    )
    master = master.replace(
        r"\input{../report/discussion.tex}",
        r"\section{Results - Part B (Bonus)}"
        + "\n"
        + r"\input{../report/part_b/results.tex}"
        + "\n"
        + r"\input{../report/part_b/discussion.tex}",
    )
    master = master.replace("../report/references.tex", "../report/part_b/references.tex")
    master = master.replace(
        r"\input{../report/failure_examples.tex}",
        r"\input{../report/failure_examples.tex}"
        + "\n"
        + r"\section{Part B before/after examples}"
        + "\n"
        + r"\label{app:partb}"
        + "\n"
        + r"\input{../report/part_b/case_examples.tex}",
    )
    master = master.replace("../report/code_appendix.tex", "../report/part_b/code_appendix.tex")
    (ROOT / "1RT730_Assignment2/assignment2_part_ab.tex").write_text(master)


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    freeze = json.loads((ROOT / "config/part_b_freeze.json").read_text())
    for run in RUNS.values():
        manifest = json.loads((run / "run.json").read_text())
        if manifest["identity"]["experiment_freeze_sha256"] != sha256(
            ROOT / "config/part_b_freeze.json"
        ):
            raise ValueError("A formal run differs from the recorded freeze.")
        if manifest["identity"]["code_sha256"] != freeze["code_sha256"]:
            raise ValueError("Formal run source differs from the frozen source.")
    result = compare_fairness(
        ROOT / "runs/part_a_baseline", list(RUNS.values()), REPORT / "comparison.json"
    )
    if result["baseline"]["metrics"]["inputs"] != 120:
        raise ValueError("The primary Part B comparison requires 120 matched inputs.")
    export_tables(result)
    export_figures(result)
    export_cases()
    export_code()
    export_combined_sources(result)
    rows = []
    for name, run in {"baseline": ROOT / "runs/part_a_baseline", **RUNS}.items():
        for row in scored_fairness(run)[0]:
            rows.append({"condition": name, **row})
    pd.DataFrame(rows).to_csv(REPORT / "scored_samples.csv", index=False)
    print("Combined report sources exported; compile assignment2_part_ab.tex.")


if __name__ == "__main__":
    main()
