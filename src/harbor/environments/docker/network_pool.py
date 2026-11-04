"""Opt-in per-trial bridge subnets, allocated atomically by Docker IPAM.

Set HARBOR_DOCKER_SUBNET_POOL to a private CIDR checked against the Docker
host's routes; HARBOR_DOCKER_SUBNET_PREFIX defaults to 28. Unset the pool
for normal Compose networking. Only simple Linux public-network tasks are
supported. No daemon restart or image rebuild is required.

Networks retained after a crash are identifiable by harbor.session and
harbor.subnet-pool labels. Remove only after confirming the owning trial
has ended; an empty network may belong to an in-progress startup.
"""

import asyncio
import ipaddress
import json
import logging
import os
import subprocess
import uuid

logger = logging.getLogger(__name__)
OWNER_LABEL = "harbor.subnet-owner"


class NetworkPool:
    def __init__(self, pool: str, prefix: int, session_id: str):
        self.pool = ipaddress.IPv4Network(pool)
        if not self.pool.is_private or not self.pool.prefixlen <= prefix <= 29:
            raise ValueError(
                "Harbor subnet pool must be private IPv4 with pool prefix <= subnet prefix <= 29"
            )
        self.count = 1 << (prefix - self.pool.prefixlen)
        if self.count > 4096:
            raise ValueError("Harbor subnet pool supports at most 4096 trial subnets")
        self.prefix = prefix
        self.owner = uuid.uuid4().hex
        self.name = f"harbor-subnet-{self.owner}"
        self.session_id = session_id
        self.subnet: str | None = None

    @classmethod
    def from_env(cls, session_id: str) -> "NetworkPool | None":
        pool = os.environ.get("HARBOR_DOCKER_SUBNET_POOL", "").strip()
        if not pool:
            return None
        return cls(
            pool, int(os.environ.get("HARBOR_DOCKER_SUBNET_PREFIX", "28")), session_id
        )

    @staticmethod
    def _docker(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=30, check=False
        )

    def _allocate(self) -> None:
        offset = int(self.owner, 16) % self.count
        size = 1 << (32 - self.prefix)
        for step in range(self.count):
            address = (
                int(self.pool.network_address) + ((offset + step) % self.count) * size
            )
            subnet = str(ipaddress.IPv4Network((address, self.prefix)))
            result = self._docker(
                "network",
                "create",
                "--driver",
                "bridge",
                "--subnet",
                subnet,
                "--label",
                f"{OWNER_LABEL}={self.owner}",
                "--label",
                f"harbor.session={self.session_id}",
                "--label",
                f"harbor.subnet-pool={self.pool}",
                self.name,
            )
            if result.returncode == 0:
                self.subnet = subnet
                return
            if (
                "pool overlaps with other one on this address space"
                not in result.stderr.lower()
            ):
                raise RuntimeError(
                    f"Cannot create Harbor network {self.name}: {result.stderr.strip()}"
                )
        raise RuntimeError(
            f"Harbor subnet pool {self.pool} exhausted ({self.count} /{self.prefix} slots)"
        )

    async def allocate(self) -> None:
        if self.subnet is not None:
            raise RuntimeError("Harbor network already allocated")
        # Let an in-flight daemon create finish before cleanup, including cancellation.
        task = asyncio.create_task(asyncio.to_thread(self._allocate))
        try:
            await asyncio.shield(task)
        except BaseException:
            try:
                await task
            except (RuntimeError, OSError, subprocess.SubprocessError):
                pass
            try:
                await self.release()
            except (RuntimeError, OSError, subprocess.SubprocessError):
                logger.exception(
                    "Failed to clean up network %s after allocation failure", self.name
                )
            raise

    def _release(self) -> None:
        result = self._docker("network", "inspect", self.name)
        if result.returncode:
            if (
                "no such network" in result.stderr.lower()
                or "not found" in result.stderr.lower()
            ):
                self.subnet = None
                return
            raise RuntimeError(
                f"Cannot inspect Harbor network {self.name}: {result.stderr.strip()}"
            )
        info = json.loads(result.stdout)[0]
        if (info.get("Labels") or {}).get(OWNER_LABEL) != self.owner:
            raise RuntimeError(
                f"Refusing to remove network with different ownership: {self.name}"
            )
        result = self._docker("network", "rm", self.name)
        if result.returncode:
            raise RuntimeError(
                f"Cannot remove Harbor network {self.name}: {result.stderr.strip()}"
            )
        self.subnet = None

    async def release(self) -> None:
        await asyncio.to_thread(self._release)

    def compose_config(self) -> dict[str, object]:
        return {"networks": {"default": {"external": True, "name": self.name}}}
