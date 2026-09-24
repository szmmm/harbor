"""Read prompt JSONL without requiring benchmark rubrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override

from harbor.models.task.paths import TaskPaths

from ddb_explore.base import ExploreAdapter


@dataclass(frozen=True)
class PromptTask:
    id: str
    instruction: str


class PromptDataset:
    def __init__(self, path: Path, prompt_field: str = "query") -> None:
        self.records: dict[str, PromptTask] = {}
        self.original_rows: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object")
            prompt = row.get(prompt_field)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Line {line_number}: nonempty {prompt_field!r} is required"
                )
            digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            task_id = str(row.get("id") or f"query-{digest}")
            if task_id in self.records:
                raise ValueError(f"Line {line_number}: duplicate task id {task_id!r}")
            self.records[task_id] = PromptTask(
                id=task_id,
                instruction=prompt,
            )
            self.original_rows[task_id] = row

    def ids(self) -> list[str]:
        return list(self.records)

    def get(self, task_id: str) -> PromptTask:
        return self.records[task_id]


class PromptAdapter(ExploreAdapter[PromptTask]):
    task_prefix = "ddb-explore"
    source = "prompt-jsonl"
    default_device = "HUAWEI Mate 80 Pro Max"

    @override
    def _write_tests(self, paths: TaskPaths, record: PromptTask) -> None:
        super()._write_tests(paths, record)
        (paths.tests_dir / "prompt_source.json").write_text(
            json.dumps(
                cast(PromptDataset, self.dataset).original_rows[record.id],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @override
    def _task_metadata(self, record: PromptTask) -> dict[str, Any]:
        return {"prompt_id": record.id}
