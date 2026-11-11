"""Unavailable previews reach the existing rubric judge, without a reward override."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "adapters/ddb_explore/src/ddb_explore/task-template/tests"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(TEMPLATE))
    helpers = load(
        ROOT / "adapters/ddb_explore/tests/test_rubric_judge.py", "preview_helpers"
    )
    explore = load(TEMPLATE / "test_explore.py", "preview_explore")
    return helpers, explore


def make_judge(tmp_path, helpers, script, *, failing_browser=True):
    class Env(helpers.FakeEnv):
        async def goto(self, url):
            if failing_browser:
                raise RuntimeError(
                    "Navigation timeout at https://preview.test/?token=SECRET"
                )
            await super().goto(url)

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_text("No application was implemented.")
    judge = helpers.RubricJudge(
        Env(),
        helpers.ProbeConfig(model="m", api_key="k", max_steps=6),
        tmp_path / "evidence",
        deadline_s=60,
        source_root=root,
        executor=helpers.FakeExecutor(),
        inspector=helpers.FakeInspector(),
    )
    client = helpers.FakeClient(script)
    judge._make_client = lambda: client
    return judge, client


def test_unavailable_preview_can_be_judged_zero_from_source(
    tmp_path, modules, monkeypatch
):
    h, _ = modules
    script = [
        [("read_source_file", {"path": "README.md"})],
        [
            (
                "record_verdicts",
                {"verdicts": [h.verdict(i, "FAIL", "e0001") for i in h.IDS]},
            )
        ],
        [("finish", {"summary": "Required app was not implemented."})],
    ]
    judge, client = make_judge(tmp_path, h, script)
    result = asyncio.run(
        judge.evaluate("http://app", h.RUBRIC, preview_error="HTTP 202: not ready")
    )
    assert result["status"] == "completed" and result["score"] == 0
    assert result["model_turns"] == 3
    assert result["preview_readiness_error"] == "HTTP 202: not ready"
    names = {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert {"read_source_file", "return_to_app", "record_verdicts", "finish"} <= names
    assert "HTTP 202: not ready" in json.dumps(client.requests[0]["messages"])
    assert "SECRET" not in json.dumps(judge.evidence)
    monkeypatch.setenv("VLM_REWARD_ENABLED", "1")
    monkeypatch.setenv("TRAJECTORY_REWARD_ENABLED", "0")
    assert h.combine_rewards.build_rewards(result, {})["reward"] == 0


@pytest.mark.parametrize("preview_error", [None, "HTTP 202: not ready"])
def test_navigation_failure_and_retry_do_not_abort_judge(
    tmp_path, modules, preview_error
):
    h, _ = modules
    judge, client = make_judge(
        tmp_path,
        h,
        [
            [("return_to_app", {})],
            [("list_source_files", {})],
            [("finish", {"summary": "Insufficient evidence to blame the app."})],
        ],
    )
    result = asyncio.run(
        judge.evaluate("http://app", h.RUBRIC, preview_error=preview_error)
    )
    assert len(client.requests) == 3
    assert result["status"] == "evaluator_failed" and result["score"] is None
    assert [e["action"] for e in judge.evidence] == [
        "open",
        "return_to_app",
        "list_source_files",
    ]
    assert [e["success"] for e in judge.evidence] == [False, False, True]


def test_recovered_preview_keeps_normal_scoring(tmp_path, modules):
    h, _ = modules
    judge, client = make_judge(
        tmp_path,
        h,
        [
            [("click_ref", {"ref": 0})],
            [
                (
                    "record_verdicts",
                    {"verdicts": [h.verdict(i, "PASS", "e0001") for i in h.IDS]},
                )
            ],
            [("finish", {"summary": "Preview recovered; all requirements checked."})],
        ],
        failing_browser=False,
    )
    result = asyncio.run(
        judge.evaluate("http://app", h.RUBRIC, preview_error="Earlier timeout")
    )
    assert result["score"] == 1 and result["status"] == "completed"
    assert len(client.requests) == 3


@pytest.mark.parametrize("url", ["http://preview", ""])
def test_readiness_failure_reaches_normal_probe(tmp_path, modules, monkeypatch, url):
    _, explore = modules
    monkeypatch.setenv("VLM_API_KEY", "test")
    monkeypatch.setenv("APP_BASE_URL", url)
    monkeypatch.setenv("VLM_EXPLORE_ENABLED", "1")
    spec = tmp_path / "app_spec.json"
    rubric = tmp_path / "rubric.json"
    spec.write_text('{"default_device":"desktop"}')
    rubric.write_text('{"prompt":"Build an app"}')
    monkeypatch.setattr(explore, "APP_SPEC_PATH", spec)
    monkeypatch.setattr(explore, "RUBRIC_PATH", rubric)
    monkeypatch.setattr(explore, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(
        explore, "wait_for_app_ready", lambda *args: "HTTP 202: not ready"
    )
    calls = []

    async def run_probe(app_url, app_spec, rubric, *, preview_error=None):
        calls.append((app_url, preview_error))
        return {"status": "completed", "score": 0}

    monkeypatch.setattr(explore, "run_probe", run_probe)
    assert explore.main() == 0
    assert len(calls) == 1 and calls[0][0] == url and calls[0][1]
    assert (
        json.loads(explore.REPORT_PATH.read_text())["preview_readiness_error"]
        == calls[0][1]
    )


def test_all_generated_tasks_have_handoff():
    dirs = [TEMPLATE, *sorted((ROOT / "datasets/ddb-explore").glob("*/tests"))]
    assert len(dirs) == 101
    for directory in dirs:
        explore = (directory / "test_explore.py").read_text()
        judge = (directory / "judge.py").read_text()
        assert '"status": "app_unavailable"' not in explore
        assert "preview_error=preview_error" in explore
        assert "preview_readiness_note" in judge
