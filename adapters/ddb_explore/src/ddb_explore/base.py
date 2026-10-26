"""Write Harbor tasks without knowledge of a source dataset."""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from harbor.models.task.config import TaskConfig
from harbor.models.task.paths import TaskPaths

LOGGER = logging.getLogger(__name__)
TEMPLATE_DIR = Path(__file__).parent / "task-template"


class ExploreRecord(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def instruction(self) -> str: ...


class ExploreDataset[Record: ExploreRecord](Protocol):
    def ids(self) -> list[str]: ...

    def get(self, task_id: str) -> Record: ...


def docker_hub_name(image: str) -> str:
    """Full Docker Hub repository path: python:3.11-slim -> library/python:3.11-slim."""
    return image if "/" in image else f"library/{image}"


def to_safe_id(raw_id: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]", "-", raw_id.lower())
    return safe if re.match(r"^[a-z0-9]", safe) else f"t-{safe}"


class ExploreAdapter[Record: ExploreRecord]:
    task_prefix: str
    source: str
    default_device: str
    verifier_env: dict[str, str] = {}

    def __init__(
        self,
        output_dir: Path,
        *,
        dataset: ExploreDataset[Record],
        template_dir: Path | None = None,
        task_ids: Iterable[str] | None = None,
        overwrite: bool = False,
        limit: int | None = None,
        registry_mirror: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.templates_dir = Path(template_dir or TEMPLATE_DIR)
        self.dataset = dataset
        self._task_ids = list(task_ids) if task_ids is not None else None
        self._overwrite = overwrite
        self._limit = limit
        self.registry_mirror = (registry_mirror or "").strip("/")

    def available_task_ids(self) -> list[str]:
        return self.dataset.ids()

    def run(self) -> list[Path]:
        return self.prepare_tasks(
            task_ids=self._task_ids, overwrite=self._overwrite, limit=self._limit
        )

    def prepare_tasks(
        self,
        *,
        output_dir: Path | None = None,
        limit: int | None = None,
        task_ids: Iterable[str] | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        selected = (
            list(dict.fromkeys(str(tid) for tid in task_ids))
            if task_ids is not None
            else self.available_task_ids()
        )
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            selected = selected[:limit]
        safe_ids = [to_safe_id(task_id) for task_id in selected]
        if len(safe_ids) != len(set(safe_ids)):
            raise ValueError("Task IDs collide after filesystem normalization")
        generated = []
        for task_id in selected:
            result = self.generate_task(task_id, overwrite=overwrite)
            if result is not None:
                generated.append(result)
        return generated

    def generate_task(self, task_id: str, *, overwrite: bool = False) -> Path | None:
        record = self.dataset.get(task_id)
        task_dir = self.output_dir / f"{self.task_prefix}-{to_safe_id(task_id)}"
        if task_dir.exists():
            if not overwrite:
                LOGGER.info("Skipping %s (already exists)", task_id)
                return None
            shutil.rmtree(task_dir)
        paths = TaskPaths(task_dir=task_dir)
        try:
            for directory in (
                paths.environment_dir,
                paths.solution_dir,
                paths.tests_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            self._write_instruction(paths, record)
            self._write_environment(paths)
            self._write_solution(paths, record)
            self._write_tests(paths, record)
            self._write_task_config(paths, record)
        except Exception:
            shutil.rmtree(task_dir, ignore_errors=True)
            raise
        return task_dir

    def _write_instruction(self, paths: TaskPaths, record: Record) -> None:
        paths.instruction_path.write_text(record.instruction, encoding="utf-8")

    def _write_environment(self, paths: TaskPaths) -> None:
        shutil.copytree(
            self.templates_dir / "environment",
            paths.environment_dir,
            dirs_exist_ok=True,
        )
        if self.registry_mirror:
            # Hosts without Docker Hub access pull the base image through a
            # mirror, as the r2egym runtime datasets do (dockerproxy.net).
            dockerfile = paths.environment_dir / "Dockerfile"
            text = dockerfile.read_text()
            dockerfile.write_text(
                re.sub(
                    r"^FROM (\S+)",
                    lambda m: f"FROM {self.registry_mirror}/{docker_hub_name(m.group(1))}",
                    text,
                    flags=re.MULTILINE,
                )
            )

    def _write_solution(self, paths: TaskPaths, record: Record) -> None:
        shutil.copy2(self.templates_dir / "solution/solve.sh", paths.solve_path)
        paths.solve_path.chmod(0o755)

    def _app_spec(self, record: Record) -> dict[str, Any]:
        return {"id": record.id, "instruction": record.instruction}

    def _device(self, record: Record) -> str:
        return self.default_device

    def _write_tests(self, paths: TaskPaths, record: Record) -> None:
        shutil.copytree(
            self.templates_dir / "tests",
            paths.tests_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (paths.tests_dir / "test.sh").chmod(0o755)
        app_spec = {**self._app_spec(record), "default_device": self._device(record)}
        encoded = json.dumps(app_spec, ensure_ascii=False, indent=2) + "\n"
        for directory in (paths.tests_dir, paths.environment_dir):
            (directory / "app_spec.json").write_text(encoded, encoding="utf-8")

    def _task_metadata(self, record: Record) -> dict[str, Any]:
        return {}

    def _write_task_config(self, paths: TaskPaths, record: Record) -> None:
        raw = (self.templates_dir / "task.toml").read_text()
        raw = raw.replace("{{TASK_ID}}", to_safe_id(record.id))
        raw = raw.replace("{{TASK_PREFIX}}", self.task_prefix)
        raw = raw.replace("{{DEFAULT_DEVICE}}", self._device(record))
        config = TaskConfig.model_validate_toml(raw)
        config.metadata = {**config.metadata, **self._task_metadata(record)}
        config.source = self.source
        config.verifier.env.update(self.verifier_env)
        paths.config_path.write_text(config.model_dump_toml())
