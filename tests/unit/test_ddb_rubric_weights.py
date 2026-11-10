"""Regression checks for weighted rubric rewards and missing evaluations."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "adapters/ddb_explore/src/ddb_explore/task-template/tests"
SCORERS = [TEMPLATE / "rubric_score.py"] + sorted(
    (ROOT / "datasets/ddb-explore").glob("*/tests/rubric_score.py")
)


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=SCORERS, ids=lambda path: str(path.relative_to(ROOT)))
def scorer(request):
    return load_module(request.param)


LEAVES = [
    {"id": "critical", "critical": True, "branch": "intention"},
    {"id": "ordinary1", "critical": False, "branch": "static"},
    {"id": "ordinary2", "critical": False, "branch": "static"},
]


@pytest.mark.parametrize(
    "verdicts, expected",
    [
        (["PASS", "PASS", "PASS"], 1),
        (["FAIL", "FAIL", "FAIL"], 0),
        (["FAIL", "PASS", "PASS"], 0.5),
        (["PASS", "FAIL", "PASS"], 0.75),
        (["PASS", "FAIL", "FAIL"], 0.5),
    ],
)
def test_critical_weight_and_failure(scorer, verdicts, expected):
    values = dict(zip([leaf["id"] for leaf in LEAVES], verdicts))
    result = scorer.score_tree(LEAVES, values, min_coverage=1)
    assert result["status"] == "completed"
    assert result["score"] == pytest.approx(expected)
    assert result["pass_rate"] == pytest.approx(verdicts.count("PASS") / 3)
    assert result["failed_critical"] == (["critical"] if verdicts[0] == "FAIL" else [])


def test_critical_failure_respects_coverage(scorer):
    result = scorer.score_tree(LEAVES, {"critical": "FAIL"}, min_coverage=0.9)
    assert result["status"] == "incomplete" and result["score"] is None


def test_unknown_is_excluded_from_weight(scorer):
    values = {"critical": "FAIL", "ordinary1": "PASS"}
    result = scorer.score_tree(LEAVES, values, min_coverage=0.5)
    assert result["score"] == pytest.approx(1 / 3)


def test_missing_critical_remains_unavailable(scorer):
    result = scorer.score_tree(LEAVES, {"ordinary1": "PASS"}, min_coverage=0)
    assert result["status"] == "incomplete" and result["score"] is None


def test_failed_critical_does_not_hide_an_unresolved_critical(scorer):
    leaves = LEAVES + [{"id": "missing", "critical": True, "branch": "dynamic"}]
    result = scorer.score_tree(leaves, {"critical": "FAIL"}, min_coverage=0)
    assert result["status"] == "incomplete" and result["score"] is None


def test_weighted_reward_reaches_training(scorer, monkeypatch):
    monkeypatch.setenv("VLM_REWARD_ENABLED", "1")
    monkeypatch.setenv("TRAJECTORY_REWARD_ENABLED", "0")
    result = scorer.score_tree(
        LEAVES,
        {"critical": "FAIL", "ordinary1": "PASS", "ordinary2": "PASS"},
        min_coverage=1,
    )
    combine = load_module(TEMPLATE / "combine_rewards.py")
    rewards = combine.build_rewards(result, {"status": "disabled"})
    assert rewards["reward"] == rewards["vlm_reward"] == 0.5
    assert rewards["vlm_critical_failed"] == 1


def test_all_dataset_tasks_are_covered():
    assert len(SCORERS) == 101
