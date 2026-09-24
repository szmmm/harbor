"""
Adapter that converts WebGen-Bench tasks into Harbor task structure.

Dataset: luzimu/WebGen-Bench on HuggingFace
Paper: https://arxiv.org/abs/2505.03733
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override

from ddb_explore.base import ExploreAdapter
from harbor.models.task.paths import TaskPaths

LOGGER = logging.getLogger(__name__)

WEBGEN_TEMPLATE_DIR = Path(__file__).parent / "task-template"
BASELINE_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[3] / "webgen_bench/src/webgen_bench/task-template"
)


@dataclass(frozen=True)
class WebGenBenchTask:
    id: str
    instruction: str
    primary_category: str
    subcategories: list[str]
    application_type: str
    ui_instruct: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WebGenBenchTask":
        if not d.get("id") or not str(d.get("instruction", "")).strip():
            raise ValueError("WebGen-Bench rows require id and instruction")
        rubrics = d.get("ui_instruct")
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError(
                "Original WebGen-Bench rows must contain ui_instruct rubrics"
            )
        for rubric in rubrics:
            if not isinstance(rubric, dict) or not all(
                isinstance(rubric.get(key), str) and rubric[key].strip()
                for key in ("task", "expected_result")
            ):
                raise ValueError("Each rubric requires task and expected_result")
        cat = d.get("Category", {})
        return cls(
            id=str(d.get("id", "")),
            instruction=str(d.get("instruction", "")),
            primary_category=str(cat.get("primary_category", "")),
            subcategories=list(cat.get("subcategories", [])),
            application_type=str(d.get("application_type", "")),
            ui_instruct=rubrics,
        )


class WebGenBenchDataset:
    """Loads WebGen-Bench records from HuggingFace or a local JSONL file."""

    def __init__(
        self,
        split: str = "test",
        jsonl_path: Path | None = None,
    ) -> None:
        self.split = split
        self.jsonl_path = jsonl_path
        self._records: dict[str, WebGenBenchTask] | None = None

    def _ensure_loaded(self) -> None:
        if self._records is not None:
            return

        records: dict[str, WebGenBenchTask] = {}

        if self.jsonl_path is not None:
            LOGGER.info("Loading WebGen-Bench from %s", self.jsonl_path)
            for line in Path(self.jsonl_path).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                task = WebGenBenchTask.from_dict(d)
                records[task.id] = task
        else:
            LOGGER.info("Loading WebGen-Bench from HuggingFace (split=%s)", self.split)
            from datasets import load_dataset

            ds = load_dataset("luzimu/WebGen-Bench", split=self.split)
            for row in ds:
                task = WebGenBenchTask.from_dict(dict(row))
                records[task.id] = task

        self._records = records
        LOGGER.info("Loaded %d WebGen-Bench tasks", len(records))

    def ids(self) -> list[str]:
        self._ensure_loaded()
        assert self._records is not None
        return sorted(self._records.keys())

    def get(self, task_id: str) -> WebGenBenchTask:
        self._ensure_loaded()
        assert self._records is not None
        try:
            return self._records[task_id]
        except KeyError as exc:
            raise KeyError(f"WebGen-Bench task {task_id!r} not found") from exc


class WebGenBenchAdapter(ExploreAdapter[WebGenBenchTask]):
    """Add original WebGen-Bench instructions, rubrics, and baseline evaluator."""

    task_prefix = "webgen-explore"
    source = "luzimu/WebGen-Bench"
    default_device = "desktop"
    verifier_env = {
        "WEBGEN_BASELINE_ENABLED": "${WEBGEN_BASELINE_ENABLED:-1}",
        "WEBGEN_BASELINE_URL": "${WEBGEN_BASELINE_URL:-}",
    }

    def __init__(
        self,
        output_dir: Path,
        *,
        dataset: WebGenBenchDataset | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(output_dir, dataset=dataset or WebGenBenchDataset(), **kwargs)

    @override
    def _write_instruction(self, paths: TaskPaths, record: WebGenBenchTask) -> None:
        template = (WEBGEN_TEMPLATE_DIR / "instruction.md").read_text()
        paths.instruction_path.write_text(
            template.replace("{{ instruction }}", record.instruction)
        )

    @override
    def _write_environment(self, paths: TaskPaths) -> None:
        dockerfile = (BASELINE_TEMPLATE_DIR / "environment/Dockerfile").read_text()
        # Install before task-specific COPY so tasks share the browser layer.
        dockerfile = dockerfile.replace(
            "WORKDIR /app",
            'RUN pip install --no-cache-dir "playwright>=1.60" '
            "&& playwright install --with-deps chromium\n\nWORKDIR /app",
            1,
        )
        (paths.environment_dir / "Dockerfile").write_text(dockerfile)

    @override
    def _write_solution(self, paths: TaskPaths, record: WebGenBenchTask) -> None:
        super()._write_solution(paths, record)
        staged_app = Path(__file__).parents[2] / "solutions" / record.id
        if staged_app.exists():
            shutil.copytree(staged_app, paths.solution_dir / "app")

    @override
    def _app_spec(self, record: WebGenBenchTask) -> dict[str, Any]:
        return {
            **super()._app_spec(record),
            "application_type": record.application_type,
            "primary_category": record.primary_category,
            "subcategories": record.subcategories,
        }

    @override
    def _write_tests(self, paths: TaskPaths, record: WebGenBenchTask) -> None:
        super()._write_tests(paths, record)
        baseline_dir = BASELINE_TEMPLATE_DIR / "tests"
        shutil.copy2(
            WEBGEN_TEMPLATE_DIR / "tests/test_baseline.py",
            paths.tests_dir / "test_baseline.py",
        )
        shutil.copy2(
            baseline_dir / "test_ui.py", paths.tests_dir / "webgen_baseline.py"
        )
        shutil.copytree(
            baseline_dir / "webvoyager",
            paths.tests_dir / "webvoyager",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        original = {**self._app_spec(record), "ui_instruct": record.ui_instruct}
        encoded = json.dumps(original, ensure_ascii=False, indent=2) + "\n"
        for directory in (paths.tests_dir, paths.environment_dir):
            (directory / "ui_instruct.json").write_text(encoded)

    @override
    def _task_metadata(self, record: WebGenBenchTask) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "webgen_bench_id": record.id,
            "application_type": record.application_type,
            "primary_category": record.primary_category,
        }
        if record.subcategories:
            metadata["subcategories"] = record.subcategories
        return metadata
