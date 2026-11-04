import asyncio
import json
import subprocess
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.docker.network_pool import OWNER_LABEL, NetworkPool
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def result(code=0, out="", err=""):
    return subprocess.CompletedProcess([], code, out, err)


def test_opt_in_and_validation(monkeypatch):
    monkeypatch.delenv("HARBOR_DOCKER_SUBNET_POOL", raising=False)
    assert NetworkPool.from_env("x") is None
    monkeypatch.setenv("HARBOR_DOCKER_SUBNET_POOL", "172.30.240.0/20")
    assert NetworkPool.from_env("x").count == 256
    for pool, prefix in [
        ("8.8.8.0/24", 28),
        ("172.30.0.0/16", 15),
        ("172.30.0.0/16", 31),
    ]:
        with pytest.raises(ValueError):
            NetworkPool(pool, prefix, "x")


@pytest.mark.asyncio
async def test_retry_overlap_and_external_network():
    pool = NetworkPool("172.30.240.0/27", 28, "x")
    pool._docker = Mock(
        side_effect=[
            result(1, err="Pool overlaps with other one on this address space"),
            result(),
        ]
    )
    await pool.allocate()
    assert pool._docker.call_count == 2
    assert (
        pool._docker.call_args_list[0].args[5] != pool._docker.call_args_list[1].args[5]
    )
    assert pool.compose_config()["networks"]["default"] == {
        "external": True,
        "name": pool.name,
    }


@pytest.mark.asyncio
async def test_exhaustion():
    pool = NetworkPool("172.30.240.0/28", 28, "x")
    pool._docker = Mock(
        return_value=result(1, err="Pool overlaps with other one on this address space")
    )
    pool.release = AsyncMock()
    with pytest.raises(RuntimeError, match="exhausted"):
        await pool.allocate()
    pool.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_permission_error_is_not_exhaustion():
    pool = NetworkPool("172.30.240.0/20", 28, "x")
    pool._docker = Mock(return_value=result(1, err="authorization denied"))
    pool.release = AsyncMock()
    with pytest.raises(RuntimeError, match="authorization denied"):
        await pool.allocate()
    assert pool._docker.call_count == 1


@pytest.mark.asyncio
async def test_cancel_waits_for_creation_then_releases():
    pool = NetworkPool("172.30.240.0/28", 28, "x")
    entered, finish = threading.Event(), threading.Event()

    def allocate():
        entered.set()
        finish.wait(5)
        pool.subnet = "172.30.240.0/28"

    pool._allocate = allocate
    pool.release = AsyncMock()
    task = asyncio.create_task(pool.allocate())
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    pool.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_release_requires_ownership():
    pool = NetworkPool("172.30.240.0/28", 28, "x")
    pool._docker = Mock(
        return_value=result(out=json.dumps([{"Labels": {OWNER_LABEL: "someone-else"}}]))
    )
    with pytest.raises(RuntimeError, match="different ownership"):
        await pool.release()
    assert pool._docker.call_count == 1


@pytest.mark.asyncio
async def test_release_and_repeat():
    pool = NetworkPool("172.30.240.0/28", 28, "x")
    pool.subnet = "172.30.240.0/28"
    pool._docker = Mock(
        side_effect=[
            result(out=json.dumps([{"Labels": {OWNER_LABEL: pool.owner}}])),
            result(),
            result(1, err="No such network"),
        ]
    )
    await pool.release()
    await pool.release()
    assert pool.subnet is None


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_DOCKER_SUBNET_POOL", "172.30.240.0/20")
    root = tmp_path / "environment"
    root.mkdir()
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n")
    paths = TrialPaths(trial_dir=tmp_path / "trial")
    paths.mkdir()
    env = DockerEnvironment(
        environment_dir=root,
        environment_name="test",
        session_id="test-pool",
        trial_paths=paths,
        task_env_config=EnvironmentConfig(docker_image="python:3.11-slim"),
    )
    env._validate_daemon_mode = Mock()
    env._validate_image_os = AsyncMock()
    env._network_pool.allocate = AsyncMock()
    env._network_pool.release = AsyncMock()
    env._run_docker_compose_command = AsyncMock()
    env.ensure_dirs = AsyncMock()
    env._upload_environment_dir_after_start = AsyncMock()
    env.prepare_logs_for_host = AsyncMock()
    return env


@pytest.mark.asyncio
async def test_start_stop_lifecycle(env):
    await env.start(False)
    path = env._network_compose_path
    assert path in env._docker_compose_paths
    assert json.loads(path.read_text())["networks"]["default"]["external"] is True
    await env.stop(True)
    env._network_pool.release.assert_awaited_once()
    assert not path.exists()


@pytest.mark.asyncio
async def test_failed_start_releases(env):
    env._upload_environment_dir_after_start.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        await env.start(False)
    env._network_pool.release.assert_awaited_once()
    assert env._network_compose_path is None


@pytest.mark.asyncio
async def test_keep_preserves_network(env):
    await env.start(False)
    env._keep_containers = True
    await env.stop(True)
    env._network_pool.release.assert_not_awaited()
    assert env._network_compose_path.exists()


def test_custom_compose_rejected(env):
    (env.environment_dir / "docker-compose.yaml").write_text("services: {}")
    with pytest.raises(ValueError, match="without custom Compose"):
        DockerEnvironment(
            environment_dir=env.environment_dir,
            environment_name="test",
            session_id="custom",
            trial_paths=env.trial_paths,
            task_env_config=EnvironmentConfig(),
        )
