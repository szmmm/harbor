"""Browser tools and model configuration shared by the rubric judge."""

from __future__ import annotations

import base64
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from browser_env import (
    MAX_NODES_CAP,
    ActionExecutor,
    Device,
    PageInspector,
    PlaywrightEnv,
)

logger = logging.getLogger(__name__)


# Steps held back from exploration so a challenged report has somewhere to go.


@dataclass
class ProbeConfig:
    model: str
    api_key: str
    base_url: str = ""
    max_steps: int = 12
    max_nodes: int = 60
    # The report_findings payload (a tour plus several findings, often in
    # Chinese) is the longest thing the model emits; 4096 truncated it.
    max_output_tokens: int = 65536
    temperature: float = 0.3
    request_timeout_s: float = 180.0
    max_retries: int = 3
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Some gateways 400 on `temperature`. Arm this up front for a gateway
    # already known to refuse it and skip the one wasted request discovery
    # would otherwise cost.
    omit_temperature: bool = False


def _fn(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


_REF = {
    "type": "integer",
    "description": "ref index [N] from the latest a11y snapshot.",
}

_SWIPE_VECTORS: dict[str, tuple[int, int]] = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}

# Roles that are a control inside a row rather than the row itself. Swiping one
# of these is almost never what was meant: the gesture goes to the widget, and
# the row it sits in does not move.
_ROW_CONTROL_ROLES = frozenset(
    {
        "combobox",
        "listbox",
        "select",
        "option",
        "button",
        "link",
        "textbox",
        "searchbox",
        "spinbutton",
        "checkbox",
        "radio",
        "switch",
        "slider",
        "menuitem",
        "tab",
    }
)


_PROBE_TOOLS: list[dict[str, Any]] = [
    _fn(
        "click_ref",
        "Click the element with this ref.",
        {
            "ref": _REF,
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "default": "left",
            },
            "clicks": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
        },
        required=["ref"],
    ),
    _fn(
        "type_into",
        "Focus the element with this ref and type text into it.",
        {"ref": _REF, "text": {"type": "string"}},
        required=["ref", "text"],
    ),
    _fn(
        "swipe_ref",
        "Swipe across the element with this ref using real touch events — the "
        "gesture behind swipe-to-reveal rows and carousels. Use this when a "
        "control is marked COVERED, or when the app's copy promises a swipe. "
        "This is the ONLY way to test a touch gesture; clicking cannot.",
        {
            "ref": _REF,
            "direction": {"type": "string", "enum": ["left", "right", "up", "down"]},
            "distance": {
                "type": "integer",
                "minimum": 20,
                "maximum": 400,
                "default": 120,
                "description": "Pixels to travel.",
            },
        },
        required=["ref", "direction"],
    ),
    _fn(
        "long_press_ref",
        "Press and hold the element with this ref using real touch events — "
        "the gesture behind 'long-press to save/edit/delete' and context "
        "menus on a phone UI. A click cannot substitute for it, so when the "
        "app's own copy promises a long press, this is the only way to test "
        "whether it works.",
        {
            "ref": _REF,
            "duration_ms": {
                "type": "integer",
                "minimum": 200,
                "maximum": 3000,
                "default": 700,
            },
        },
        required=["ref"],
    ),
    _fn(
        "upload_file",
        "Pick a file for the next file picker, then click the element with "
        "this ref to open it. Use this on any 'upload' / 'choose image' / "
        "'import' control: clicking it with `click_ref` leaves a modal picker "
        "open that nothing answers, which makes every later step time out and "
        "reads as the app hanging.",
        {
            "ref": _REF,
            "kind": {
                "type": "string",
                "enum": ["image", "text", "csv", "json"],
                "default": "image",
                "description": "What kind of small valid file to hand it. A "
                "generated fixture; you do not supply bytes.",
            },
        },
        required=["ref"],
    ),
    _fn(
        "press_key",
        "Press a key or combo, e.g. 'Enter', 'Tab', 'Ctrl+S'.",
        {"key": {"type": "string"}},
        required=["key"],
    ),
    _fn(
        "scroll",
        "Scroll the viewport, then re-read the refreshed snapshot.",
        {
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "amount": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
        },
        required=["direction"],
    ),
    _fn(
        "wait",
        "Wait for the UI to settle, then re-read the snapshot.",
        {
            "duration_ms": {
                "type": "integer",
                "minimum": 50,
                "maximum": 10000,
                "default": 500,
            }
        },
        required=["duration_ms"],
    ),
    _fn("reload", "Reload the page.", {}),
    _fn(
        "get_page_info",
        "Read the page's visible TEXT — counters, totals, messages, "
        "empty-state copy. The a11y snapshot lists interactive elements only, "
        "so this is the only way to see body text.",
        {},
    ),
    _fn(
        "get_a11y_snapshot",
        "Re-read the snapshot with a bigger budget or a folded role expanded. "
        "The automatic per-step snapshot uses the default budget, so this is "
        "the ONLY way to see nodes beyond it.",
        {
            "max_nodes": {"type": "integer", "minimum": 5, "maximum": MAX_NODES_CAP},
            "expand_roles": {"type": "array", "items": {"type": "string"}},
        },
    ),
    _fn(
        "locate_text",
        "Measure where a piece of text is actually drawn and which box "
        "encloses it. Use this for EVERY claim about which element a label, "
        "badge, tag or price belongs to, and for any left/right/above/below "
        "relationship. `get_page_info` returns document order, in which a "
        "floating badge appears next to a DIFFERENT card than the one it is "
        "drawn on — that mistake has been made and cost two rounds.",
        {
            "text": {
                "type": "string",
                "description": "Exact substring to locate, e.g. '最划算'.",
            }
        },
        required=["text"],
    ),
    _fn(
        "take_screenshot",
        "See the current viewport as pixels. Use for colour, layout, spacing, "
        "canvas graphics, and to confirm anything visual before you report it.",
        {},
    ),
]


# Gestures that count as having acted on the page. Kept apart from
# _ACTION_TOOLS, which is also what `verified_by="interaction"` is checked
# against, so widening one does not silently widen the other.


def _same_element(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two snapshot nodes describe the same control.

    Role and name only. `value` is excluded on purpose: typing into a field
    changes it, and that is exactly the case this check has to survive.
    """
    return (
        str(a.get("role") or "").lower() == str(b.get("role") or "").lower()
        and (a.get("name") or "").strip() == (b.get("name") or "").strip()
    )


class ProbeAgent:
    """One a11y-first probing loop over one app."""

    def __init__(
        self,
        env: PlaywrightEnv,
        config: ProbeConfig,
        executor: ActionExecutor | None = None,
        inspector: PageInspector | None = None,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._env = env
        self._cfg = config
        self._executor = executor or ActionExecutor(env)
        self._inspector = inspector or PageInspector(env)
        self._nodes: list[dict[str, Any]] = []
        self._expand_roles: list[str] = []
        self._max_nodes = max(5, min(config.max_nodes, MAX_NODES_CAP))
        self._no_temperature = config.omit_temperature
        self._trace_sink = trace_sink
        # How many cumulative runtime entries have already been shown to the
        # model, so each step gets only what its own actions produced.
        self._runtime_shown: dict[str, int] = {}

    def _emit(self, event: dict[str, Any]) -> None:
        """Hand one trajectory event to the sink as it happens.

        The round JSON is only written once the run finishes, so an interrupted
        or crashed probe used to leave nothing behind — including the steps that
        did complete. A sink that appends and flushes per event is what makes
        the trajectory survive a kill, and what makes it readable while the
        probe is still walking the app.
        """
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(event)
        except Exception as e:  # noqa: BLE001 — recording must never fail a run
            logger.warning("trace sink dropped an event: %s", e)

    # ── tools ──────────────────────────────────────────────────────

    def _runtime_delta_note(self) -> str:
        """Report console/network activity produced since the last step.

        The probe used to be blind to this: the runtime snapshot was read once,
        after the loop, so a click that fired a 500 looked to the agent exactly
        like a click that did nothing, and the two conclusions it can draw from
        that — "the button is dead" and "saving fails on the server" — are
        different change requests. Sending only the delta keeps the agent from
        re-reading a failure it already accounted for.
        """
        try:
            snap = self._env.runtime_signals_snapshot()
        except Exception as e:  # noqa: BLE001 — never fail a step over telemetry
            logger.debug("runtime snapshot unavailable: %s", e)
            return ""
        lines: list[str] = []
        for channel, render in (
            ("console_errors", lambda e: f"console: {e.get('text') or ''}"),
            (
                "failed_requests",
                lambda e: (
                    f"network {e.get('kind') or ''} "
                    f"{e.get('method') or ''} "
                    f"{('HTTP ' + str(e.get('status'))) if e.get('status') else ''} "
                    f"{e.get('url') or ''}"
                    f"{' :: ' + str(e.get('error')) if e.get('error') else ''}"
                ),
            ),
        ):
            entries = snap.get(channel) or []
            seen = self._runtime_shown.get(channel, 0)
            fresh = entries[seen:]
            self._runtime_shown[channel] = len(entries)
            lines.extend(render(e) for e in fresh[:6])
            if len(fresh) > 6:
                lines.append(f"... and {len(fresh) - 6} more in {channel}")
        if not lines:
            return ""
        body = "\n".join(f"- {_truncate_line(line)}" for line in lines)
        return (
            "\n\n## Runtime activity since your last step\n"
            "The browser recorded this while your actions ran. A failed "
            "request here is what a control that 'did nothing' actually "
            "did, and reporting it as a dead control asks for the wrong "
            "fix. An error unrelated to what you touched is still worth a "
            "finding of its own.\n" + body
        )

    def _snapshot_args(self) -> dict[str, Any]:
        return {"max_nodes": self._max_nodes, "expand_roles": self._expand_roles}

    def _resolve_ref(self, ref: Any) -> dict[str, Any] | None:
        try:
            idx = int(ref)
        except (TypeError, ValueError):
            return None
        return self._nodes[idx] if 0 <= idx < len(self._nodes) else None

    async def _resolve_ref_live(self, ref: Any) -> dict[str, Any] | None:
        """Resolve ``ref``, re-reading coordinates from a fresh snapshot.

        The model batches several actions into one step, but the snapshot only
        refreshed at step boundaries — so every action after the first in a
        batch used coordinates captured before the earlier ones ran. Filling
        three form fields in one step landed the third click on the second
        field, appending "09:00–17:00"'s digits to the value already there, and
        the probe reported the "21700" it had produced itself as an app defect
        for three rounds running.

        The ref index belongs to the snapshot the model was shown, so the fresh
        copy is only trusted where it still describes the same element. If the
        page changed enough that it does not, the stale coordinates are the
        better guess: acting on the right element at the wrong place beats
        acting on whatever now sits at that index.
        """
        stale = self._resolve_ref(ref)
        if stale is None:
            return None
        snap = await self._inspector.interactive_snapshot(self._snapshot_args())
        if not snap.get("success"):
            return stale
        fresh_nodes = (snap.get("data") or {}).get("nodes", [])
        self._nodes = fresh_nodes
        fresh = self._resolve_ref(ref)
        if fresh is None or not _same_element(stale, fresh):
            return stale
        return fresh

    async def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in ("click_ref", "type_into"):
            node = await self._resolve_ref_live(args.get("ref"))
            if node is None:
                return {
                    "success": False,
                    "error": f"ref {args.get('ref')!r} is not in the "
                    f"current snapshot; re-read it",
                }
            if node.get("occluded"):
                # Clicking would land on whatever covers it, the page would not
                # change, and the probe would report a working control as dead.
                # This is how a swipe-to-delete button parked behind its own row
                # got reported as a broken delete for three rounds running.
                return {
                    "success": False,
                    "error": f"ref {args.get('ref')!r} "
                    f"({node.get('name')!r}) is covered by another "
                    f"element, so a click would hit the cover, not "
                    f"it. Reveal it first — `swipe_ref` across the "
                    f"row if this is a swipe-to-reveal action — "
                    f"then act on the fresh snapshot. Do NOT report "
                    f"this as an unresponsive control.",
                }
            xy = {"x": node.get("x"), "y": node.get("y")}
            if name == "click_ref":
                out = await self._executor.click(
                    {
                        **xy,
                        "button": args.get("button", "left"),
                        "clicks": args.get("clicks", 1),
                    }
                )
                await self._env.wait_settled()
                return out
            # Triple-click selects the field's current value so typing replaces
            # it. A single click only places a caret, so re-typing a field the
            # probe had already filled concatenated the two values — and the
            # probe then reported its own "30000500" as an app defect.
            await self._executor.click({**xy, "clicks": 3})
            out = await self._executor.type_text({"text": args.get("text", "")})
            # Typing can start work of its own — validation, autocomplete, a
            # debounced search — and reading the snapshot mid-flight shows a
            # field that has not caught up with what was typed into it.
            await self._env.wait_settled()
            return out

        if name == "press_key":
            out = await self._executor.press_key({"key": args.get("key", "")})
            await self._env.wait_settled()
            return out
        if name == "swipe_ref":
            # Occlusion is not a reason to refuse here: the point of a swipe on
            # a covered node is to drag the thing covering it out of the way.
            node = await self._resolve_ref_live(args.get("ref"))
            if node is None:
                return {
                    "success": False,
                    "error": f"ref {args.get('ref')!r} is not in the "
                    f"current snapshot; re-read it",
                }
            dx, dy = _SWIPE_VECTORS.get(str(args.get("direction")), (0, 0))
            if (dx, dy) == (0, 0):
                return {
                    "success": False,
                    "error": f"unknown direction {args.get('direction')!r}",
                }
            distance = max(20, min(int(args.get("distance") or 120), 400))
            x, y = int(node.get("x") or 0), int(node.get("y") or 0)
            out = await self._executor.touch_swipe(
                {
                    "start_x": x,
                    "start_y": y,
                    "end_x": x + dx * distance,
                    "end_y": y + dy * distance,
                }
            )
            await self._env.wait_settled()
            # Name what was actually swiped. A swipe-to-reveal row is swiped by
            # its body, but the refs on that row are its controls, and a gesture
            # aimed at one of those moves the control rather than the row — a
            # left-swipe that landed on a day-picker combobox was read as the
            # app having no swipe-to-delete, and two rounds went to a feature
            # that worked. Silence here left nothing to notice the miss by.
            if out.get("success"):
                out = {
                    **out,
                    "text": _swipe_outcome_text(
                        out, node, str(args.get("direction")), distance
                    ),
                }
            return out
        if name == "long_press_ref":
            node = await self._resolve_ref_live(args.get("ref"))
            if node is None:
                return {
                    "success": False,
                    "error": f"ref {args.get('ref')!r} is not in the "
                    f"current snapshot; re-read it",
                }
            out = await self._executor.touch_long_press(
                {
                    "x": node.get("x"),
                    "y": node.get("y"),
                    "duration_ms": args.get("duration_ms") or 700,
                }
            )
            await self._env.wait_settled()
            return out
        if name == "upload_file":
            node = await self._resolve_ref_live(args.get("ref"))
            if node is None:
                return {
                    "success": False,
                    "error": f"ref {args.get('ref')!r} is not in the "
                    f"current snapshot; re-read it",
                }
            path = _upload_fixture(str(args.get("kind") or "image"))
            # Arm before the click: the chooser opens synchronously with it, and
            # an unanswered chooser is modal — every later action then times out
            # against a dialog the agent cannot see, which reads as the app
            # hanging rather than as the probe having asked a question.
            self._env.arm_file_upload([str(path)])
            out = await self._executor.click({"x": node.get("x"), "y": node.get("y")})
            await self._env.wait_settled()
            if out.get("success"):
                out = {
                    **out,
                    "text": (
                        f"clicked {node.get('name')!r} with "
                        f"{path.name} armed for the file picker"
                    ),
                }
            return out
        if name == "scroll":
            return await self._executor.scroll(args)
        if name == "wait":
            return await self._executor.wait(args)
        if name == "reload":
            return await self._executor.reload(args)
        if name == "get_page_info":
            return await self._inspector.get_page_info(args)
        if name == "locate_text":
            return await self._inspector.locate_text(args)
        if name == "get_a11y_snapshot":
            requested = args.get("max_nodes")
            if requested is not None:
                self._max_nodes = max(5, min(int(requested), MAX_NODES_CAP))
            roles = args.get("expand_roles") or []
            # Sticky: once the agent asks to see gridcells unfolded, every
            # later snapshot should keep showing them.
            for role in roles:
                if role not in self._expand_roles:
                    self._expand_roles.append(str(role))
            snap = await self._inspector.interactive_snapshot(self._snapshot_args())
            self._nodes = (
                (snap.get("data") or {}).get("nodes", []) if snap.get("success") else []
            )
            return snap

        return {"success": False, "error": f"unknown tool {name!r}"}

    # ── model transport ────────────────────────────────────────────

    def _make_client(self):
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": self._cfg.api_key,
            "base_url": self._cfg.base_url or None,
            # A gateway that accepts the connection but never answers will
            # otherwise hang the whole run; the client retries on timeout.
            "timeout": self._cfg.request_timeout_s,
            "max_retries": self._cfg.max_retries,
        }
        if self._cfg.extra_headers:
            kwargs["default_headers"] = dict(self._cfg.extra_headers)
        return OpenAI(**kwargs)


def _temperature_rejected(exc: Exception) -> bool:
    text = str(exc).lower()
    return "temperature" in text and (
        "unsupported" in text
        or "deprecated" in text
        or "does not support" in text
        or "not supported" in text
    )


_UPLOAD_FIXTURES: dict[str, tuple[str, bytes]] = {
    # 1x1 red PNG. Small enough that a size limit is not what gets tested, and
    # a real decodable image so a preview that stays blank is the app's doing.
    "image": (
        "probe-upload.png",
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8"
            "DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
        ),
    ),
    "text": ("probe-upload.txt", b"probe upload fixture\n"),
    "csv": ("probe-upload.csv", b"name,value\nprobe,1\n"),
    "json": ("probe-upload.json", b'{"probe": true}\n'),
}


def _upload_fixture(kind: str) -> Path:
    """Write a small valid file of ``kind`` to a temp dir and return its path.

    Generated rather than shipped so the probe can offer an upload path without
    the repo carrying binary fixtures, and cached per kind so repeated uploads
    in one round reuse the same file.
    """
    name, body = _UPLOAD_FIXTURES.get(kind) or _UPLOAD_FIXTURES["image"]
    path = Path(tempfile.gettempdir()) / name
    if not path.exists() or path.stat().st_size != len(body):
        path.write_bytes(body)
    return path


def _truncate_line(text: str, n: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _device_block(device: Device | None) -> str:
    """Tell the probe which screen it is holding, and what that makes suspect.

    Without this the probe judges a 327x327 cover screen against the same
    expectations as a 425x949 slab and reports the narrow frame itself as the
    defect ("the layout is cramped"), which is a change request the author
    cannot act on. Naming the device turns the same observation into the useful
    one: this layout does not adapt to this screen.
    """
    if device is None:
        return ""
    lines = [
        "",
        "## The screen you are holding",
        f"{device.label}, {device.width}x{device.height} CSS px "
        f"(physical panel {device.physical}).",
        "All coordinates, screenshots and element boxes are in these CSS "
        "pixels, so a width of "
        f"{device.width} is the whole screen.",
    ]
    hint = _FORM_FACTOR_HINTS.get(device.form_factor, "")
    if hint:
        lines.append(hint)
    what = (
        "a real screen an app has to work on"
        if device.is_tablet_class
        else "a real phone screen"
    )
    lines.append(
        f"This is {what}, not a window you should judge as small. "
        "Report a layout problem when content is clipped, overlapping, "
        "unreachable or unreadable AT THIS SIZE — that is a genuine adaptation "
        "defect. Do not report the screen size itself, and do not ask the "
        "author to design for a different one."
    )
    return "\n".join(lines) + "\n"


# Per-form-factor guidance. Each names the specific mistake that shape invites,
# because "check the layout" produces nothing a probe was not already doing.
_FORM_FACTOR_HINTS = {
    "bar": "",
    "cover": (
        "This is a foldable's COVER screen — the small outer display, used "
        "shut. It is unusually short, so vertical clipping and a bottom bar "
        "that eats the content are the things to look for. A modal or form "
        "taller than the screen with no way to scroll to its submit button is "
        "a real bug here."
    ),
    "wide_fold": (
        "This is a wide-fold phone OPENED: it is much closer to square than a "
        "normal phone (nearly as wide as it is tall). Layouts built on the "
        "assumption of a tall narrow screen break here in a specific way — "
        "a hero image sized in viewport height swallows the page, a bottom "
        "sheet covers everything, a single column leaves half the width "
        "empty. Judge whether the extra width is used or wasted."
    ),
    "book_fold": (
        "This is a book-style foldable OPENED, so the screen is tablet-sized "
        "and nearly square. A phone layout stretched to this width with one "
        "column of content and vast empty margins is a real adaptation "
        "finding, as is text that now runs the full width and is hard to "
        "read. Check whether anything reflows into the space at all."
    ),
    "tri_fold": (
        "This is a tri-fold. At this width the thing to check is whether the "
        "layout treats it as a phone or a tablet, and whether that choice is "
        "the right one — a narrow fold that overflows a fixed-width element, "
        "or a wide fold that leaves most of the screen blank, are both real "
        "findings."
    ),
}


# Words that make a claim depend on where things are painted rather than what
# the DOM contains. Flattened text cannot settle any of these.
def _swipe_outcome_text(
    result: dict[str, Any],
    node: dict[str, Any],
    direction: str,
    distance: int,
) -> str:
    """Say which element the gesture was applied to, and warn if it was a widget.

    The executor reports only that the touch sequence was delivered, which reads
    as "the swipe happened" no matter what it happened to. Naming the target lets
    a miss be seen as a miss instead of as the app failing to respond.
    """
    role = str(node.get("role") or "?").lower()
    name = str(node.get("name") or "").strip()
    label = f"{role} {name!r}" if name else role
    text = (
        f"swiped {direction} {distance}px on {label} @({node.get('x')},{node.get('y')})"
    )
    if result.get("text"):
        text = f"{text}; {result['text']}"
    if role in _ROW_CONTROL_ROLES:
        text += (
            f"\nNOTE: a {role} is a control inside a row, not the row. "
            f"Swiping it moves the control, so an unchanged layout says "
            f"nothing about whether the row supports the gesture. To test "
            f"swipe-to-reveal, aim at the row body — the image, title or "
            f"other non-interactive part of the row — and if no ref names "
            f"it, take a screenshot to find the row and swipe the ref "
            f"nearest its centre."
        )
    return text
