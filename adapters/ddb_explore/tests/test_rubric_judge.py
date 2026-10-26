import asyncio
import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

import combine_rewards
from judge import FINAL_TURNS, RubricJudge, check_verdict
from probe import ProbeConfig
from rubric_score import flatten, score_tree


def leaf(leaf_id, description, *, critical=False, verify_by="view", note=""):
    return {
        "id": leaf_id,
        "description": description,
        "critical": critical,
        "origin": "explicit",
        "covers": ["R1"],
        "verify_by": verify_by,
        "note": note,
        "children": None,
    }


TREE = {
    "intention": {
        "id": "I",
        "description": "core",
        "children": [leaf("I.1", "能添加食材", critical=True, verify_by="interaction")],
    },
    "content": {"id": "C", "description": "content", "children": []},
    "static": {
        "id": "S",
        "description": "look",
        "children": [
            {
                "id": "S.1",
                "description": "card",
                "children": [leaf("S.1.1", "显示名称"), leaf("S.1.2", "显示日期")],
            }
        ],
    },
    "dynamic": {
        "id": "D",
        "description": "dynamic",
        "children": [
            {"id": "D.B", "description": "basic", "children": [leaf("D.B.1", "可删除", verify_by="interaction")]},
            {"id": "D.C", "description": "complex", "children": [leaf("D.C.1", "刷新后仍在", verify_by="interaction")]},
        ],
    },
}
RUBRIC = {
    "id": "abc123",
    "prompt": "做一个食材过期提醒应用",
    "as_of": "2026-10-24",
    "method": "full",
    "spec": {"delivery": "single_page", "ui_language": "zh", "time_frame": {"date": "2026-10-24"}},
    "tree": TREE,
    "meta": {"stats": {"leaves": 5}},
}
LEAVES = flatten(TREE)
IDS = [l["id"] for l in LEAVES]


def test_flatten_reads_all_branches_in_order():
    assert IDS == ["I.1", "S.1.1", "S.1.2", "D.B.1", "D.C.1"]
    assert [l["branch"] for l in LEAVES] == ["intention", "static", "static", "dynamic", "dynamic"]
    assert LEAVES[0]["critical"] and not LEAVES[1]["critical"]


def test_score_is_flat_pass_rate():
    result = score_tree(LEAVES, {i: "PASS" for i in IDS} | {"S.1.2": "FAIL"}, min_coverage=1)
    assert result["status"] == "completed"
    assert result["score"] == pytest.approx(4 / 5)
    assert result["branches"]["static"] == pytest.approx(0.5)
    assert result["branches"]["content"] is None


def test_failed_critical_leaf_gates_to_zero_even_with_gaps():
    result = score_tree(LEAVES, {"I.1": "FAIL"}, min_coverage=0.9)
    assert (result["status"], result["score"], result["failed_critical"]) == ("completed", 0.0, ["I.1"])


def test_untested_critical_leaf_has_no_score():
    verdicts = {i: "PASS" for i in IDS} | {"I.1": "UNTESTED"}
    result = score_tree(LEAVES, verdicts, min_coverage=0)
    assert result["status"] == "incomplete" and result["score"] is None


def test_low_coverage_has_no_score_and_enough_coverage_scores_resolved_leaves():
    verdicts = {"I.1": "PASS", "S.1.1": "PASS", "S.1.2": "FAIL", "D.B.1": "PASS"}
    assert score_tree(LEAVES, verdicts, min_coverage=0.9)["score"] is None
    result = score_tree(LEAVES, verdicts, min_coverage=0.8)
    assert result["score"] == pytest.approx(3 / 4)
    assert result["counts"] == {"PASS": 3, "FAIL": 1, "UNTESTED": 1}


@pytest.mark.skipif(
    not Path(os.environ.get("RUBRICS_JSONL", "/nonexistent")).is_file(),
    reason="set RUBRICS_JSONL to check real trees",
)
def test_flatten_matches_generator_leaf_counts():
    for line in Path(os.environ["RUBRICS_JSONL"]).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert len(flatten(row["tree"])) == row["meta"]["stats"]["leaves"]


EVIDENCE = {
    "e0000": {"id": "e0000", "action": "open", "success": True},
    "e0001": {"id": "e0001", "action": "click_ref", "success": True},
    "e0002": {"id": "e0002", "action": "click_ref", "success": False},
}
BY_ID = {l["id"]: l for l in LEAVES}


@pytest.mark.parametrize(
    "item, message",
    [
        ({"leaf_id": "X", "verdict": "PASS", "reason": "r", "evidence_ids": ["e0000"]}, "unknown leaf"),
        ({"leaf_id": "S.1.1", "verdict": "MAYBE", "reason": "r", "evidence_ids": []}, "verdict must be"),
        ({"leaf_id": "S.1.1", "verdict": "PASS", "reason": " ", "evidence_ids": ["e0000"]}, "reason is empty"),
        ({"leaf_id": "S.1.1", "verdict": "PASS", "reason": "r", "evidence_ids": ["e9"]}, "unknown evidence"),
        ({"leaf_id": "S.1.1", "verdict": "FAIL", "reason": "r", "evidence_ids": []}, "at least one"),
        ({"leaf_id": "S.1.1", "verdict": "PASS", "reason": "r", "evidence_ids": ["e0002"]}, "tool succeeded"),
        ({"leaf_id": "I.1", "verdict": "PASS", "reason": "r", "evidence_ids": ["e0000"]}, "after an action"),
    ],
)
def test_check_verdict_rejects(item, message):
    with pytest.raises(ValueError, match=message):
        check_verdict(item, BY_ID, EVIDENCE)


def test_check_verdict_accepts():
    ok = check_verdict(
        {"leaf_id": "I.1", "verdict": "PASS", "reason": "added", "evidence_ids": ["e0001"]},
        BY_ID,
        EVIDENCE,
    )
    assert ok["verdict"] == "PASS"
    # A missing feature can fail from the landing page; UNTESTED needs no evidence.
    check_verdict({"leaf_id": "I.1", "verdict": "FAIL", "reason": "no form", "evidence_ids": ["e0000"]}, BY_ID, EVIDENCE)
    check_verdict({"leaf_id": "D.C.1", "verdict": "UNTESTED", "reason": "budget", "evidence_ids": []}, BY_ID, EVIDENCE)


def test_source_tree_reads_inside_root_only(tmp_path):
    from source_tools import SourceTree

    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.js").write_text("import React from 'react';\nconst x = 1;\n")
    (tmp_path / "node_modules/react").mkdir(parents=True)
    (tmp_path / "node_modules/react/index.js").write_text("react")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\0\0data")
    tree = SourceTree(tmp_path)
    listing = tree.list_files()["text"]
    assert "src/app.js" in listing and "node_modules" not in listing
    assert tree.search("from 'react'")["text"] == "src/app.js:1: import React from 'react';"
    assert "const x" in tree.read("src/app.js")["text"]
    assert not tree.read("../outside.txt")["success"]
    assert not tree.read("logo.png")["success"]
    assert not tree.search("(")["success"]
    assert not SourceTree(tmp_path / "missing").list_files()["success"]


class FakeEnv:
    async def goto(self, url):
        self.url = url

    async def set_viewport(self, width, height):
        self.viewport = (width, height)

    async def clear_site_data(self):
        self.cleared = True

    async def set_media(self, media):
        self.media = media

    def reset_runtime_log(self):
        pass

    async def screenshot_bytes(self):
        return b"\x89PNG fake"

    def runtime_signals_snapshot(self):
        return {}

    def drain_side_effects(self):
        return []

    async def wait_settled(self):
        pass


class FakeInspector:
    async def interactive_snapshot(self, args):
        return {"success": True, "text": "[0] button '添加'", "data": {"nodes": [{"name": "添加", "role": "button", "x": 5, "y": 5}]}}

    async def get_page_info(self, args):
        return {"success": True, "text": "牛奶 2026-10-30"}


class FakeExecutor:
    async def click(self, args):
        return {"success": True}

    async def reload(self, args):
        return {"success": True}


class FakeClient:
    """Answers with scripted tool calls and records each request."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=str)))
        calls = self.script.pop(0) if self.script else []
        tool_calls = [
            SimpleNamespace(
                id=f"c{len(self.requests)}-{n}",
                function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
                model_dump=lambda name=name, args=args, n=n: {
                    "id": f"c{len(self.requests)}-{n}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                },
            )
            for n, (name, args) in enumerate(calls)
        ]
        message = SimpleNamespace(content=None, tool_calls=tool_calls)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    def close(self):
        pass


def verdict(leaf_id, value, *evidence):
    return {"leaf_id": leaf_id, "verdict": value, "reason": "seen", "evidence_ids": list(evidence)}


def run_judge(tmp_path, script, *, max_steps=6, min_coverage=0.9):
    client = FakeClient(script)
    (tmp_path / "workspace").mkdir(exist_ok=True)
    (tmp_path / "workspace/index.html").write_text("<script>localStorage.setItem('x', 1)</script>")
    judge = RubricJudge(
        FakeEnv(),
        ProbeConfig(model="m", api_key="k", max_steps=max_steps, omit_temperature=True),
        tmp_path / "evidence",
        deadline_s=600,
        source_root=tmp_path / "workspace",
        executor=FakeExecutor(),
        inspector=FakeInspector(),
    )
    judge._make_client = lambda: client
    result = asyncio.run(judge.evaluate("http://app", RUBRIC, min_coverage=min_coverage))
    return result, client


def test_page_and_source_tools_record_evidence(tmp_path):
    script = [
        [("set_viewport", {"width": 1920, "height": 1080}), ("clear_site_data", {}),
         ("search_source", {"pattern": "localStorage"}), ("set_media", {"media": "print"})],
        [("record_verdicts", {"verdicts": [
            verdict("I.1", "PASS", "e0001"),
            verdict("D.C.1", "FAIL", "e0002", "e0003"),
        ] + [verdict(i, "PASS", "e0000") for i in ("S.1.1", "S.1.2", "D.B.1")]})],
        [("finish", {"summary": "done"})],
    ]
    result, client = run_judge(tmp_path, script)
    actions = [row["action"] for row in result["evidence"]]
    assert actions == ["open", "set_viewport", "clear_site_data", "search_source", "set_media"]
    assert result["evidence"][3]["screenshot"] is None
    # D.B.1 is an interaction leaf: a PASS citing only the landing page is rejected.
    assert result["leaves"][3]["verdict"] == "UNTESTED"
    assert result["status"] == "incomplete" and result["coverage"] == pytest.approx(4 / 5)
    source = json.loads((tmp_path / "evidence/e0003.json").read_text())
    assert "index.html:1:" in source["outcome"]["text"]


def test_judge_loop_scores_recorded_verdicts(tmp_path):
    script = [
        [("click_ref", {"ref": 0}), ("get_page_info", {})],
        [("record_verdicts", {"verdicts": [
            verdict("I.1", "PASS", "e0001"),
            verdict("S.1.1", "PASS", "e0002"),
            verdict("S.1.2", "FAIL", "e0002"),
            verdict("D.B.1", "PASS", "e0001"),
        ]})],
        [("finish", {"summary": "too early"})],
        [("reload", {}), ("record_verdicts", {"verdicts": [verdict("D.C.1", "PASS", "e0003")]})],
        [("finish", {"summary": "食材应用"})],
    ]
    result, client = run_judge(tmp_path, script)
    assert result["status"] == "completed", result
    assert result["score"] == pytest.approx(4 / 5)
    assert result["finished_by_model"] and result["summary"] == "食材应用"
    assert [row["verdict"] for row in result["leaves"]] == ["PASS", "PASS", "FAIL", "PASS", "PASS"]
    assert len(result["evidence"]) == 4 and (tmp_path / "evidence/e0003.png").is_file()
    # The early finish was refused with the leaves that still need a verdict.
    refused = [m for m in client.requests[3]["messages"] if m["role"] == "tool"][-1]
    assert "D.C.1" in refused["content"]
    task = json.loads(client.requests[0]["messages"][1]["content"].split("\n")[0])
    assert task["user_today"] == "2026-10-24" and task["leaves"][0]["critical"] is True


def test_judge_forces_verdicts_when_budget_ends(tmp_path):
    script = [
        [("click_ref", {"ref": 0})],
        [("click_ref", {"ref": 0})],
        [("record_verdicts", {"verdicts": [verdict(i, "PASS", "e0001") for i in IDS[:4]]
                              + [verdict("D.C.1", "UNTESTED")]})],
    ]
    result, client = run_judge(tmp_path, script, max_steps=2, min_coverage=0.8)
    final = client.requests[-1]
    assert final["tool_choice"]["function"]["name"] == "record_verdicts"
    assert [t["function"]["name"] for t in final["tools"]] == ["record_verdicts"]
    assert len(client.requests) == 3
    assert result["status"] == "completed" and result["score"] == 1.0
    assert not result["finished_by_model"]


def test_judge_without_verdicts_has_no_score(tmp_path):
    result, client = run_judge(tmp_path, [], max_steps=1)
    assert len(client.requests) == 1 + FINAL_TURNS
    assert result["status"] == "incomplete" and result["score"] is None


def test_old_observations_lose_snapshots(tmp_path):
    script = [[("click_ref", {"ref": 0})] for _ in range(7)]
    result, client = run_judge(tmp_path, script, max_steps=7)
    texts = [
        json.loads(item["text"])
        for m in client.requests[-1]["messages"]
        if m["role"] == "user" and isinstance(m["content"], list)
        for item in m["content"]
        if item["type"] == "text" and item["text"].startswith("{")
    ]
    # The open page and 7 clicks: the 4 most recent keep their snapshots.
    assert [("snapshot" in t) for t in texts] == [False] * 4 + [True] * 4
    images = [
        item for m in client.requests[-1]["messages"] if isinstance(m["content"], list)
        for item in m["content"] if item["type"] == "image_url"
    ]
    assert len(images) == 3


def test_reward_combiner_adds_rubric_diagnostics(monkeypatch):
    monkeypatch.setenv("VLM_REWARD_ENABLED", "1")
    monkeypatch.setenv("TRAJECTORY_REWARD_ENABLED", "0")
    probe = {"status": "completed", "score": 0.8, "coverage": 1.0, "pass_rate": 0.8,
             "failed_critical": [], "branches": {"intention": 1.0, "content": None}}
    rewards = combine_rewards.build_rewards(probe, {"status": "missing"}, {})
    assert rewards == {"vlm_reward": 0.8, "reward": 0.8, "vlm_coverage": 1.0, "vlm_pass_rate": 0.8,
                       "vlm_critical_failed": 0.0, "vlm_branch_intention": 1.0}


def test_adapter_writes_rubric_tasks(tmp_path):
    from ddb_explore.adapter import RubricAdapter, RubricDataset

    mobile = {**RUBRIC, "id": "Mob_1", "spec": {**RUBRIC["spec"], "delivery": "mobile_app"}}
    source = tmp_path / "rubrics.jsonl"
    source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in (RUBRIC, mobile)), encoding="utf-8")
    adapter = RubricAdapter(tmp_path / "out", dataset=RubricDataset(source), registry_mirror="dockerproxy.net/")
    dirs = adapter.run()
    dockerfile = (dirs[0] / "environment/Dockerfile").read_text()
    assert "FROM dockerproxy.net/library/python:3.11-slim\n" in dockerfile
    # One environment for all tasks, so Harbor builds and caches one image.
    assert sorted(p.name for p in (dirs[0] / "environment").iterdir()) == ["Dockerfile"]
    assert [d.name for d in dirs] == ["ddb-explore-abc123", "ddb-explore-mob_1"]
    task = dirs[0]
    assert (task / "instruction.md").read_text(encoding="utf-8") == RUBRIC["prompt"]
    assert json.loads((task / "tests/rubric.json").read_text(encoding="utf-8")) == RUBRIC
    assert json.loads((task / "tests/app_spec.json").read_text())["default_device"] == "desktop"
    assert json.loads((dirs[1] / "tests/app_spec.json").read_text())["default_device"] == "HUAWEI Mate 80 Pro Max"
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["verifier"]["env"]["VLM_EXPLORE_DEVICE"] == "${VLM_EXPLORE_DEVICE:-desktop}"
    assert config["metadata"]["prompt_id"] == "abc123"
    assert (task / "tests/rubric_score.py").is_file() and (task / "tests/judge.py").is_file()
    rows = adapter.prompt_rows(["abc123", "Mob_1"])
    assert rows[1] == {"instruction": RUBRIC["prompt"], "metadata": {"task_id": "ddb-explore-mob_1"}}
