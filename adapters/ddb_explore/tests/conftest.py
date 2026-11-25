"""Make the verifier modules importable without a browser.

The verifier files run inside the task container, where playwright exists.
These tests drive the judge with a fake browser, so a stub module is enough.
"""

import sys
import types
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ADAPTER / "src/ddb_explore/task-template/tests"), str(ADAPTER / "src")]

try:
    import playwright.async_api  # noqa: F401
except ImportError:
    api = types.ModuleType("playwright.async_api")
    for name in ("async_playwright", "Page", "Browser", "BrowserContext"):
        setattr(api, name, object)
    sys.modules["playwright"] = types.ModuleType("playwright")
    sys.modules["playwright.async_api"] = api
