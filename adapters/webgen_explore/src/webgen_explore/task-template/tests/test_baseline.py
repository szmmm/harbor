"""Run the original WebGen-Bench rubric and evaluator as a comparison metric."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from test_explore import APP_READY_TIMEOUT, start_dev_server, wait_for_app_ready

REPORT = Path("/logs/verifier/webgen-report.json")


def main() -> None:
    import webgen_baseline as baseline

    process = None
    result = {"status": "evaluator_failed", "score": None}
    try:
        data = json.loads(Path("/tests/ui_instruct.json").read_text())
        url = os.environ.get("WEBGEN_BASELINE_URL") or os.environ.get("APP_BASE_URL")
        if not url:
            started = start_dev_server()
            if started is None:
                raise RuntimeError("App dev server did not start")
            process, url = started
        if not wait_for_app_ready(url, APP_READY_TIMEOUT):
            raise RuntimeError("Preview readiness timed out")
        # Preserve original tasks, prompt, budget, and scoring. Redirect output.
        baseline.REWARD_PATH = REPORT.parent / "webgen-reward.txt"
        baseline.run_webvoyager(baseline.create_tasks_jsonl(data, url))
        score = baseline.compute_score(data)
        stats = json.loads((REPORT.parent / "category_stats.json").read_text())
        result = {
            "status": "completed"
            if not stats["counts"]["start_failed"]
            else "incomplete",
            "score": score,
            "details": stats,
            "app_url": url,
            "state_isolation": "fresh browser sessions; shared backend is not reset",
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        results_dir = baseline.WORK_DIR / "results"
        if results_dir.exists():
            shutil.copytree(
                results_dir, REPORT.parent / "webvoyager_results", dirs_exist_ok=True
            )
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
