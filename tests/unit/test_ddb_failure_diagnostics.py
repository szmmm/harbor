import asyncio
import json
from unittest.mock import AsyncMock

from harbor.agents.installed.ddb import DDB


def test_failure_extracts_event_error_and_captures_proxy_progress(tmp_path):
    agent = object.__new__(DDB)
    agent.logs_dir = tmp_path
    agent._api_key = "private-key"
    agent.proxy_url = "http://proxy"
    agent._get_session = AsyncMock(
        return_value={
            "turns": [
                {
                    "agent_id": "ddt_agent1",
                    "completion_token_ids": [1, 2],
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    task = {
        "task_id": "task",
        "status": "failed",
        "result": None,
        "events": [
            {
                "type": "error",
                "data": {
                    "iteration": 11,
                    "message": "provider failed",
                    "retryable": True,
                },
            },
            {
                "type": "complete",
                "data": {"error": "Provider failed Bearer private-key"},
            },
        ],
    }
    failure = asyncio.run(agent._capture_failure(task, "session"))
    assert failure["proxy_progress"]["completed_turns"] == 1
    assert failure["proxy_progress"]["completion_tokens"] == 2
    saved = (tmp_path / "failure.json").read_text()
    assert "private-key" not in saved and "Provider failed" in failure["error"]
    assert json.loads(saved)["events"][0]["retryable"] is True
