# Assignment 2: local LLM trustworthiness evaluation

Evaluate `google/gemma-4-e4b` through LM Studio using three TrustLLM subsets:

- **Fairness:** stereotype recognition.
- **Robustness:** paired original and modified AdvGLUE inputs.
- **Machine ethics:** moral action judgments from ETHICS.

The model is served locally. This Python project downloads data, builds samples,
records model responses, and scores structured classifications against gold labels.
It does not load model weights or call a hosted inference provider.

## Git checkpoints

The private backup repository is
[`cai7874/gemma-trustllm-evaluation`](https://github.com/cai7874/gemma-trustllm-evaluation).
It includes the implementation, lockfile, configuration, frozen sample manifests,
raw model responses, scoring exports, report sources, and the compiled Part A PDF.
The environment, model weights, caches, credentials, and downloaded raw datasets are
excluded. Dataset access remains a prerequisite for downloading the full source data.

- `main` contains the completed, verified Part A checkpoint.
- The annotated `part-a` tag permanently identifies that checkpoint.
- Part B planning, implementation, and validation take place on `part-b`.
- Once Part B is completed and validated, merge it into `main` with a merge commit,
  then create the annotated `part-b` tag on the merged commit. Do not tag an unfinished
  experiment as complete.

To inspect Part A while Part B is in progress, use `git show part-a:README.md`
or browse the `part-a` tag on GitHub. Switching to `part-a` gives a detached HEAD;
switch back to `part-b` before making development changes.

## Environment

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Dependencies are defined in
`pyproject.toml` and locked in `uv.lock`. The project environment lives in `.venv`.

```sh
uv sync --locked
uv run a2 --help
```

The initial environment was created using the desktop's bundled Python 3.12.14.
On another machine, uv can obtain the Python version specified in `.python-version`.
Do not copy `.venv`; rebuild it from the lockfile instead.

Load the model in LM Studio and enable its local server. Update `config/part_a.toml`
if the model ID or server address differs. Requests require structured JSON output.
The configured `reasoning_effort = "none"` disables thinking for the verified endpoint.
There are no tools, retrieval, or shared conversation histories in the requests.

Request authentication, if needed, uses `LM_STUDIO_API_KEY`. The Hugging Face client
uses its existing login or `HF_TOKEN`. Never place credentials in project files.

## Development smoke test

Run the following from the project root, in order:

```sh
uv run a2 check
uv run a2 prepare --profile smoke --output data/prepared/smoke
uv run a2 generate --prepared data/prepared/smoke --run runs/smoke
uv run a2 score --run runs/smoke
uv run a2 report --run runs/smoke
```

`check` verifies the loaded model and downloads the three JSON files from the pinned
Hugging Face revision. TrustLLM is gated: accept the dataset access conditions on
Hugging Face before downloading it. The raw data has 1,000 fairness inputs, 912
AdvGLUE inputs (456 complete pairs), and 500 ETHICS inputs.

`prepare` selects disjoint development and reserved test samples using seed 44.
The smoke profile uses only development samples: one per fairness class, one per
ETHICS class, and one complete pair per AdvGLUE task. This gives **13 inputs**.
The saved manifest identifies the source revision, file hashes, sampling settings,
and reserved formal test IDs. Smoke scores only verify the pipeline; they are not
benchmark conclusions and do not meet the assignment's sample-size requirements.

Use a new output directory when repeating preparation or starting another experiment.
Resume an interrupted, compatible generation with:

```sh
uv run a2 generate --prepared data/prepared/smoke --run runs/smoke --resume
```

Completed inputs are not requested again. Resume checks the samples, model settings,
model metadata, prompt protocol, and Python source hashes. Recorded technical failures
must be inspected; use a fresh run after resolving them. Each request has at most two
attempts, restricted to connection errors, timeouts, HTTP 429, or server errors. Model
mistakes, invalid JSON, abstentions, and truncated answers are not retried.

## Formal Part A

The full profile reserves 120 fairness inputs, 120 ETHICS inputs, and 30 original/
modified pairs per AdvGLUE task: **480 inputs in total**. Classification sampling is
balanced by gold class, and robustness sampling is balanced by task. Fairness topic
coverage is reported rather than assumed to match the full dataset distribution.

```sh
uv run a2 prepare --profile full --output data/prepared/part_a
uv run a2 generate --prepared data/prepared/part_a --run runs/part_a_baseline
uv run a2 score --run runs/part_a_baseline
uv run a2 report --run runs/part_a_baseline
```

Run these only when the formal experiment is authorized. The smoke test does not
execute them. Part B can reuse the formal paired sample list and frozen baseline.

The formal run was authorized on 11 October 2026. Its inference source was archived
in `runs/part_a_baseline/source_snapshot/` and checked against the run's source hashes.
AdvGLUE includes repeated base texts: the selected 240 rows contain 211 unique prompts
and 91 unique original prompts. Four exact prompts also occur in the development
pool. The split separates attack-pair IDs; it is not a complete base-text holdout.
No prompt tuning was performed on these repeats. Fairness and ETHICS have no exact
prompt overlap with development.

## Scoring protocol

The original benchmark prompt is retained as the user message. A shared system
message requests English JSON (`label`, `explanation`) instead of the dataset's
legacy number-only or keyword-only output instructions. Allowed labels depend on
the task. `abstain` is available when the model declines or cannot decide.

Gold labels and other source metadata are never sent to the model. Parsing uses JSON
and schema checks, with no keyword or regex inference from explanations.

- **Fairness:** accuracy, macro-F1, per-class recall, and a confusion matrix.
- **Machine ethics:** accuracy, balanced accuracy, macro-F1, and per-class recall.
- **Robustness:** original accuracy, perturbed accuracy, accuracy drop, and attack
  success rate (ASR), both pooled and per task. ASR is the proportion of originally
  correct cases that become incorrect after perturbation. A zero denominator is N/A.
- Abstentions and invalid or truncated responses remain in the denominator as
  incorrect answers. Technical failures block scoring instead of being treated as
  model errors. Balanced accuracy is the mean of the two reference-label recalls.
  The ETHICS file includes commonsense and justice topics; pooled two-class metrics
  do not reproduce the official topic-based ETHICS aggregate.

This is a **TrustLLM subset evaluation with an adapted structured-output protocol**.
It does not reproduce the original full benchmark or its keyword/GPT scoring pipeline.
Do not directly equate these results with published leaderboard numbers. The brief
explanation field also changes the original output protocol and must be reported.

## Outputs and report

Each run contains `run.json`, `samples.json`, and append-only `responses.jsonl`.
Scoring adds `scores.json`, `scored_samples.json`, and `scored_samples.csv`.
Report export creates English `report.md`, `metrics.csv`, `metrics.tex`, PNG figures,
and all incorrect cases in `failure_cases.json`. Scoring and export are entirely
offline and do not generate new model responses.

For the final submission, use `1RT730_Assignment2/assignment2.tex`. Include essential
results and at least five real failure examples per selected subset. Review cases
manually, and show both original and perturbed responses for robustness examples.
The export reports a shortfall; it does not fabricate cases or automatically run more
inputs. Any extension of the formal sample must be planned and disclosed.

Include **all** Python modules in the code appendix, or provide a repository link.
The template's placeholder `your_code.py` alone is not the implementation. Cite the
model, LM Studio, TrustLLM, ETHICS, StereoSet, AdvGLUE, and relevant software, and add
an accurate generative-AI use statement. The basic export produces report material. The complete report follows the supplied
template in `1RT730_Assignment2/assignment2.tex`, with the template's permitted
`listings` alternative to `minted`, avoiding a shell-escape requirement. Its author
is Junmo Cai. The original template is preserved in `report/template_original.tex`.
After scoring, the following offline command exports tables and five manually
selected, audited failures per dimension:

```sh
uv run python scripts/build_part_a_report.py
```

Failure selections and interpretations are in `report/failure_selection.json`;
the exporter checks each selected failure against the saved scores and copies the
actual prompts and responses verbatim. Compile with a standard LaTeX environment or
Overleaf, or use the project-local Tectonic 0.17.0 binary installed for this report:

```sh
cd 1RT730_Assignment2
XDG_CACHE_HOME="$PWD/../.cache/tectonic/cache" ../.cache/tectonic/tectonic --keep-logs --outdir ../output/pdf assignment2.tex
```

The final PDF is `output/pdf/assignment2.pdf`. The complete Python implementation,
offline report exporter, evaluation tests, and configuration are included
in its appendix. No repository publication is required. The baseline contains no
Part B mitigation.

### ETHICS data audit and primary results

The initial 480-input run used the mixed commonsense/justice file exactly as supplied.
A source audit matched 249 of 250 justice rows to the original author's pinned data.
Justice uses reasonable/unreasonable claims; its numeric labels have incompatible
semantics with the TrustLLM copy's generic wrong/not-wrong prompt and labels. The
initial mixed-topic score is preserved as a diagnostic, rather than presented as
clean moral-action accuracy.

A documented refinement evaluates 120 **commonsense-only** examples, 60 per class,
using seed 44 and excluding all original development IDs. No labels, model settings,
or prompts are changed. Selection is independent of individual model correctness.
Twenty-six examples overlap the initial test and were requested again. The primary
report therefore uses 120 fairness inputs and 240 robustness inputs from the initial
run, plus 120 inputs from the commonsense refinement: 480 primary inputs and 600
recorded generations in total. No prompt tuning or model mitigation is involved.

```sh
uv run python scripts/audit_ethics_labels.py --download
uv run python scripts/ethics_commonsense.py prepare
uv run a2 generate --prepared data/prepared/ethics_commonsense --run runs/ethics_commonsense
uv run python scripts/ethics_commonsense.py score
uv run python scripts/build_part_a_report.py
```

`report/primary_scores.json` and `report/primary_scored_samples.csv` contain the
primary results. `runs/part_a_baseline/scores.json` remains the initial mixed-topic
experiment. The audit and both response logs are retained for transparent review.

## Code layout and checks

- `data.py`: dataset provenance, validation, and sampling.
- `tasks.py`: task labels, messages, output schema, and answer validation.
- `client.py`: local inference and bounded technical retries.
- `experiment.py`: run identity, persistence, and resume.
- `analysis.py`: offline metrics and report material.
- `io.py`: JSON files and hashing.
- `cli.py`: explicit command orchestration.

```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
```

Offline tests cover the task-specific gold mappings, disjoint/reproducible sampling,
pair completeness, ASR's conditional denominator, abstentions, invalid/truncated
outputs, gold-label leakage, technical failures, and duplicate resume records.

## Sources

- [TrustLLM dataset](https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset)
- [TrustLLM code and paper](https://github.com/HowieHwong/TrustLLM)
- [Gemma 4 E4B model card](https://huggingface.co/google/gemma-4-E4B-it)
- [LM Studio structured output](https://lmstudio.ai/docs/developer/openai-compat/structured-output)
