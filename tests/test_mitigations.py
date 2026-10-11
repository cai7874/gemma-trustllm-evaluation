"""Check paired comparisons, review provenance, and unchanged default inference."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from a2.analysis import score_response
from a2.client import LMStudioClient
from a2.comparison import paired_changes
from a2.io import write_json, write_jsonl
from a2.mitigations import development_split, load_parent, mitigation_messages
from a2.tasks import messages


def sample(sample_id="one", context="A context"):
    return {
        "id": sample_id,
        "task": "stereotype_recognition",
        "dimension": "fairness",
        "prompt": "Classify this benchmark input.",
        "gold_label": "SECRET_REFERENCE",
        "source_record": {"label": "SECRET_REFERENCE", "context": context},
    }


def record(item, label="unrelated"):
    return {
        "sample_id": item["id"],
        "status": "completed",
        "messages": messages(item),
        "elapsed_seconds": 1.0,
        "response": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps({"label": label, "explanation": "Initial answer."})
                    },
                }
            ],
        },
    }


def test_mitigation_messages_do_not_leak_reference_and_keep_prompt():
    item = sample()
    for strategy in ("definition_template", "self_review"):
        sent = mitigation_messages(item, strategy, record(item))
        assert sent[1] == {"role": "user", "content": item["prompt"]}
        assert "SECRET_REFERENCE" not in json.dumps(sent)
    assert mitigation_messages(item, "baseline") == messages(item)


def test_review_uses_actual_parent_answer_and_rejects_wrong_id():
    item = sample()
    parent = record(item)
    sent = mitigation_messages(item, "self_review", parent)
    assert sent[2]["content"] == parent["response"]["choices"][0]["message"]["content"]
    parent["sample_id"] = "other"
    with pytest.raises(ValueError, match="mismatched"):
        mitigation_messages(item, "self_review", parent)


def test_context_overlap_is_removed_even_when_ids_and_prompts_differ():
    test = sample("test", "Shared context")
    development = sample("dev", "Shared context")
    development["prompt"] = "A different continuation."
    kept, excluded = development_split([test], [development])
    assert kept == [] and excluded == ["dev"]


def test_parent_loading_validates_full_sample_and_protocol(tmp_path):
    item = sample()
    settings = {"model_id": "local-model"}
    write_json(tmp_path / "run.json", {"identity": {"config": {"model": settings}}})
    write_json(tmp_path / "samples.json", [item])
    parent = record(item)
    write_jsonl(tmp_path / "responses.jsonl", [parent])
    assert load_parent(tmp_path, [item], settings)["one"] == parent
    changed = deepcopy(item)
    changed["prompt"] = "A changed benchmark prompt."
    with pytest.raises(ValueError, match="sample differs"):
        load_parent(tmp_path, [changed], settings)
    parent["messages"][0]["content"] = "A different protocol."
    write_jsonl(tmp_path / "responses.jsonl", [parent])
    with pytest.raises(ValueError, match="prompt differs"):
        load_parent(tmp_path, [item], settings)


def test_paired_changes_join_by_id_and_measure_regressions():
    before = [
        {**sample("one"), "prediction": "unrelated", "correct": False},
        {**sample("two"), "prediction": "stereotype", "correct": True},
    ]
    after = [
        {**sample("two"), "prediction": "unrelated", "correct": False},
        {**sample("one"), "prediction": "anti-stereotype", "correct": True},
    ]
    changes = paired_changes(before, after)
    assert changes["wrong_to_correct"] == changes["correct_to_wrong"] == 1
    assert changes["label_changes"] == 2 and changes["accuracy_delta"] == 0
    after[0]["gold_label"] = "Changed reference"
    with pytest.raises(ValueError, match="reference labels changed"):
        paired_changes(before, after)
    with pytest.raises(ValueError, match="same unique"):
        paired_changes(before, before[:1])


def test_abstaining_review_is_scored_without_baseline_fallback():
    item = sample()
    item["gold_label"] = "unrelated"
    assert score_response(item, record(item))["correct"]
    reviewed = score_response(item, record(item, "abstain"))
    assert not reviewed["correct"] and reviewed["answer_status"] == "abstained"


def test_client_preserves_default_and_accepts_explicit_messages():
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(model_dump=lambda: {"choices": []})

    client = LMStudioClient.__new__(LMStudioClient)
    client.settings = {
        "max_attempts": 1,
        "model_id": "local-model",
        "temperature": 0,
        "max_tokens": 192,
        "reasoning_effort": "none",
    }
    client.api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    item = sample()
    client.generate(item)
    assert captured[-1]["messages"] == messages(item)
    custom = mitigation_messages(item, "definition_template")
    client.generate(item, prompt_messages=custom)
    assert captured[-1]["messages"] == custom
    assert captured[0]["response_format"] == captured[1]["response_format"]


def test_few_shot_only_uses_development_labels_and_rejects_overlap():
    item = sample("test", "Test context")
    example = sample("demo", "Development context")
    example["gold_label"] = "anti-stereotype"
    sent = mitigation_messages(item, "few_shot", demonstrations=[example])
    assert sent[-1] == {"role": "user", "content": item["prompt"]}
    assert json.loads(sent[2]["content"])["label"] == "anti-stereotype"
    assert "SECRET_REFERENCE" not in json.dumps(sent)
    with pytest.raises(ValueError, match="overlaps"):
        mitigation_messages(item, "few_shot", demonstrations=[item])


def test_formal_generation_requires_a_frozen_protocol(tmp_path, monkeypatch):
    from a2.mitigations import validate_freeze

    monkeypatch.chdir(tmp_path)
    config = {"part_b": {"freeze_path": "freeze.json"}}
    with pytest.raises(ValueError, match="Freeze development"):
        validate_freeze(config, {}, "few_shot")
    write_json(tmp_path / "freeze.json", {"selected_strategies": ["few_shot"], "config": {}})
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_freeze(config, {}, "few_shot")
