"""Offline checks of logic that could change experimental conclusions."""

import json

import pytest

from a2.analysis import classification_metrics, robustness_metrics, score_response
from a2.data import robustness_pairs, split_classification
from a2.experiment import validate_existing
from a2.tasks import ROBUSTNESS_LABELS, messages, parse_answer


def test_task_specific_label_encodings():
    assert ROBUSTNESS_LABELS["qnli"][0] == "yes"
    assert ROBUSTNESS_LABELS["qqp"][0] == "no"
    assert ROBUSTNESS_LABELS["mnli"] == {0: "yes", 1: "maybe", 2: "no"}


def test_explanation_does_not_determine_the_prediction():
    content = json.dumps({"label": "not wrong", "explanation": "This is not wrong, not wrong."})
    assert parse_answer(content, "implicit_ETHICS", "stop")["label"] == "not wrong"


def test_incomplete_and_out_of_schema_answers_are_not_accepted():
    content = json.dumps({"label": "wrong", "explanation": "A complete JSON object."})
    assert parse_answer(content, "implicit_ETHICS", "length")["status"] == "incomplete"
    assert (
        parse_answer('{"label":"yes","explanation":""}', "implicit_ETHICS", "stop")["status"]
        == "invalid_schema"
    )
    assert parse_answer("```json\n{}\n```", "implicit_ETHICS", "stop")["status"] == "invalid_json"


def test_gold_label_is_not_sent_to_model():
    sample = {
        "task": "implicit_ETHICS",
        "prompt": "Judge this action.",
        "gold_label": "SECRET_GOLD_LABEL",
        "source_record": {"label": "SECRET_GOLD_LABEL"},
    }
    assert "SECRET_GOLD_LABEL" not in json.dumps(messages(sample))


def test_abstentions_stay_in_accuracy_denominator():
    rows = [
        {"gold_label": "wrong", "prediction": "wrong", "correct": True, "answer_status": "valid"},
        {
            "gold_label": "not wrong",
            "prediction": "abstain",
            "correct": False,
            "answer_status": "abstained",
        },
    ]
    result = classification_metrics(rows, ("wrong", "not wrong"))
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert sum(map(sum, result["confusion_matrix"])) == 2


def pair_rows(outcomes):
    return [
        {"pair_id": str(index), "variant": variant, "correct": outcome, "answer_status": "valid"}
        for index, pair in enumerate(outcomes)
        for variant, outcome in zip(("original", "modified"), pair, strict=True)
    ]


def test_asr_is_conditional_on_original_correctness():
    result = robustness_metrics(pair_rows([(True, False), (True, True), (False, False)]))
    assert result["attack_success_rate"] == 0.5
    assert result["asr_denominator"] == 2
    assert result["original_accuracy"] == pytest.approx(2 / 3)


def test_asr_is_undefined_when_no_original_answer_is_correct():
    assert robustness_metrics(pair_rows([(False, False)]))["attack_success_rate"] is None


def test_incomplete_pair_is_rejected():
    with pytest.raises(ValueError, match="Missing scored partner"):
        robustness_metrics(pair_rows([(True, False)])[:1])
    with pytest.raises(ValueError, match="Incomplete or duplicated pair"):
        robustness_pairs(
            [{"task": "qnli", "index": 1, "type": "original", "prompt": "Question", "label": 0}]
        )


def test_split_is_reproducible_and_disjoint():
    rows = [
        {"prompt": f"Action {index}", "label": "wrong" if index < 15 else "not wrong"}
        for index in range(30)
    ]
    settings = {"ethics_test": 10, "development_per_label": 3}
    development, test = split_classification(rows, "ethics", settings, 44)
    assert len(test) == 10
    assert not {row["id"] for row in development} & {row["id"] for row in test}
    assert (development, test) == split_classification(rows, "ethics", settings, 44)


def test_duplicate_or_foreign_resume_records_are_rejected():
    samples = [{"id": "one"}]
    with pytest.raises(ValueError, match="Duplicate responses"):
        validate_existing([{"sample_id": "one"}, {"sample_id": "one"}], samples)
    with pytest.raises(ValueError, match="outside"):
        validate_existing([{"sample_id": "two"}], samples)


def test_technical_failure_is_not_a_model_mistake():
    with pytest.raises(ValueError, match="technical failure"):
        score_response({"id": "one"}, {"status": "request_failed"})
