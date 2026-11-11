"""Validate Harbor's source transfer before exposing actor output to the judge."""

import hashlib
import json
from pathlib import Path


def load_handoff(
    root: Path,
    path: Path = Path("/tmp/harbor_ddb_handoff.json"),
    receipt_path: Path = Path("/tmp/harbor_workspace_transfer.json"),
) -> dict | None:
    # Older/non-DDB trials have no transfer receipt. They retain normal grading,
    # but their source completeness must not be presented as verified.
    if not path.exists():
        return None
    handoff = json.loads(path.read_text())
    if (
        handoff.get("schema_version") != 1
        or handoff.get("execution", {}).get("status") != "completed"
    ):
        raise ValueError("DDB execution has not completed")
    if handoff.get("source_transfer", {}).get("complete") is not True:
        raise ValueError("DDB source transfer has not completed")
    receipt = json.loads(receipt_path.read_text())
    expected = receipt.get("sha256")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("complete") is not True
        or not isinstance(expected, dict)
    ):
        raise ValueError("Missing or invalid source transfer receipt")
    if not root.is_dir():
        raise ValueError("Transferred workspace directory is missing")
    actual = {}
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("Unexpected symlink in transferred workspace")
        if item.is_file():
            actual[item.relative_to(root).as_posix()] = hashlib.sha256(
                item.read_bytes()
            ).hexdigest()
    if actual != expected:
        raise ValueError("Judge workspace does not match the completed source transfer")
    if receipt.get("files_expected") != len(actual) or receipt.get(
        "files_downloaded"
    ) != len(actual):
        raise ValueError("Source transfer counts do not match the receipt")
    handoff["source_transfer"]["sandbox_hashes_verified"] = True
    handoff["source_transfer"]["inventory_complete"] = (
        receipt.get("inventory_complete") is True
    )
    return handoff
