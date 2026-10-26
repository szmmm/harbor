"""Read rubric trees from rubrics-tree-generation and write one task per tree."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override

from harbor.models.task.paths import TaskPaths

from ddb_explore.base import ExploreAdapter, to_safe_id

BRANCHES = ("intention", "content", "static", "dynamic")
PHONE = "HUAWEI Mate 80 Pro Max"
# spec.delivery -> judge screen. "unspecified" keeps the DDB phone default.
DELIVERY_DEVICES = {"mobile_app": PHONE, "single_page": "desktop", "unspecified": PHONE}


@dataclass(frozen=True)
class RubricTask:
    id: str
    instruction: str
    rubric: dict[str, Any]


class RubricDataset:
    """Rows of a rubrics.jsonl: {id, prompt, as_of, method, spec, tree, meta}."""

    def __init__(self, path: Path) -> None:
        self.records: dict[str, RubricTask] = {}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object")
            task_id, prompt = row.get("id"), row.get("prompt")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"Line {line_number}: nonempty 'id' is required")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Line {line_number}: nonempty 'prompt' is required")
            tree = row.get("tree")
            if not isinstance(tree, dict) or any(b not in tree for b in BRANCHES):
                raise ValueError(f"Line {line_number}: 'tree' needs {BRANCHES}")
            if task_id in self.records:
                raise ValueError(f"Line {line_number}: duplicate task id {task_id!r}")
            self.records[task_id] = RubricTask(id=task_id, instruction=prompt, rubric=row)

    def ids(self) -> list[str]:
        return list(self.records)

    def get(self, task_id: str) -> RubricTask:
        return self.records[task_id]


class RubricAdapter(ExploreAdapter[RubricTask]):
    task_prefix = "ddb-explore"
    source = "rubrics-tree-generation"
    default_device = PHONE

    @override
    def _device(self, record: RubricTask) -> str:
        delivery = (record.rubric.get("spec") or {}).get("delivery")
        return DELIVERY_DEVICES.get(delivery, self.default_device)

    @override
    def _write_tests(self, paths: TaskPaths, record: RubricTask) -> None:
        super()._write_tests(paths, record)
        (paths.tests_dir / "rubric.json").write_text(
            json.dumps(record.rubric, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @override
    def _task_metadata(self, record: RubricTask) -> dict[str, Any]:
        meta = record.rubric.get("meta") or {}
        values = {
            "prompt_id": record.id,
            "rubric_method": record.rubric.get("method"),
            "rubric_as_of": record.rubric.get("as_of"),
            "rubric_leaves": (meta.get("stats") or {}).get("leaves"),
        }
        return {key: value for key, value in values.items() if value is not None}

    def prompt_rows(self, task_ids: list[str]) -> list[dict[str, Any]]:
        """Prompt data rows for the slime trainer. The bridge finds a task by its directory name."""
        dataset = cast(RubricDataset, self.dataset)
        return [
            {
                "instruction": dataset.get(task_id).instruction,
                "metadata": {"task_id": f"{self.task_prefix}-{to_safe_id(task_id)}"},
            }
            for task_id in task_ids
        ]
