"""Score per-leaf verdicts on a rubric tree from rubrics-tree-generation.

Gate, then flat rate, as in the generator's reference scorer (rubric_tree/score.py):
a failed critical leaf sets the score to 0; else the score is the pass rate,
and every leaf has the same weight. Branches only group leaves.

The judge can leave a leaf UNTESTED when it could not reach the check. Such a
leaf has no verdict. A critical leaf without a verdict, or coverage below
``min_coverage``, makes the score unavailable: a missing measurement must not
become a training reward. Else the pass rate is over the leaves with a verdict.
With ``min_coverage=1`` this is the reference scorer exactly.
"""

from __future__ import annotations

from typing import Any

BRANCHES = ("intention", "content", "static", "dynamic")
VERDICTS = ("PASS", "FAIL", "UNTESTED")


def flatten(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Leaves of all branches in tree order. A node is a leaf if children is None."""
    out: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], branch: str) -> None:
        children = node.get("children")
        if children is None:
            out.append(
                {
                    "id": node["id"],
                    "branch": branch,
                    "description": node["description"],
                    "note": node.get("note") or "",
                    "verify_by": node.get("verify_by") or "view",
                    "critical": node.get("critical") is True,
                }
            )
            return
        for child in children:
            walk(child, branch)

    for branch in BRANCHES:
        root = tree.get(branch)
        if root is None:
            raise ValueError(f"Rubric tree has no {branch!r} branch")
        for child in root.get("children") or []:
            walk(child, branch)
    ids = [leaf["id"] for leaf in out]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Rubric tree needs leaves with unique ids")
    return out


def score_tree(
    leaves: list[dict[str, Any]],
    verdicts: dict[str, str],
    *,
    min_coverage: float,
) -> dict[str, Any]:
    """``verdicts`` maps leaf id to PASS, FAIL or UNTESTED; absent means UNTESTED."""
    resolved = {
        leaf["id"]: verdicts[leaf["id"]] == "PASS"
        for leaf in leaves
        if verdicts.get(leaf["id"]) in {"PASS", "FAIL"}
    }
    failed_critical = [
        leaf["id"]
        for leaf in leaves
        if leaf["critical"] and resolved.get(leaf["id"]) is False
    ]
    unresolved_critical = [
        leaf["id"] for leaf in leaves if leaf["critical"] and leaf["id"] not in resolved
    ]
    coverage = len(resolved) / len(leaves)
    n_pass = sum(resolved.values())
    branches = {}
    for branch in BRANCHES:
        results = [resolved[l["id"]] for l in leaves if l["branch"] == branch and l["id"] in resolved]
        branches[branch] = sum(results) / len(results) if results else None
    result = {
        "coverage": coverage,
        "pass_rate": n_pass / len(resolved) if resolved else None,
        "branches": branches,
        "failed_critical": failed_critical,
        "unresolved_critical": unresolved_critical,
        "counts": {
            verdict: sum(1 for l in leaves if verdicts.get(l["id"], "UNTESTED") == verdict)
            for verdict in VERDICTS
        },
    }
    if failed_critical:
        return {**result, "status": "completed", "score": 0.0}
    if unresolved_critical:
        return {**result, "status": "incomplete", "score": None,
                "error": f"Critical leaves without a verdict: {unresolved_critical}"}
    if coverage < min_coverage:
        return {**result, "status": "incomplete", "score": None,
                "error": f"Coverage {coverage:.2f} is below {min_coverage:.2f}"}
    return {**result, "status": "completed", "score": result["pass_rate"]}
