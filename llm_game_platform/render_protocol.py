"""Generic 2D scene protocol.

A generated environment's ``render()`` returns a *scene*: plain JSON data that a
trusted renderer draws. The renderer never needs to know the game rules, and the
scene is never fed to the learner (observations are a separate channel).

Scene format::

    {"width": 300, "height": 300, "background": "#101721",   # width/height required
     "items": [ ... ], "hud": "optional status text"}

Item types (all coordinates are in scene units, origin top-left, y grows down):

    rect     x y w h            (x, y = top-left)       radius?
    circle   x y r              (x, y = centre)
    line     x1 y1 x2 y2
    polygon  points=[[x, y], ...]
    text     x y text           size? anchor? ("start"|"middle"|"end")

Common optional fields: fill, stroke, stroke_width, opacity (0..1),
rotation (degrees, about the item's own centre / anchor), effect
("none"|"pulse"|"fade"|"shake"), id, and ``action`` (an int: clicking the item
sends that action for the human-controlled agent).
"""

from __future__ import annotations

import math
import re
from typing import Any

MAX_ITEMS = 2000
MAX_TEXT = 200
MAX_COORD = 1e5
_COLOR = re.compile(r"^(#[0-9a-fA-F]{3,8}|[a-zA-Z]{3,20})$")
_ITEM_FIELDS = {
    "rect": ("x", "y", "w", "h"),
    "circle": ("x", "y", "r"),
    "line": ("x1", "y1", "x2", "y2"),
    "polygon": (),
    "text": ("x", "y"),
}
_OPTIONAL_NUMBERS = ("stroke_width", "opacity", "rotation", "radius", "size")
_EFFECTS = {"none", "pulse", "fade", "shake"}
_ANCHORS = {"start", "middle", "end"}


class SceneError(ValueError):
    pass


def _number(value: Any, name: str, lo: float = -MAX_COORD, hi: float = MAX_COORD) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < lo or value > hi:
        raise SceneError(f"{name} must be finite and within [{lo:g}, {hi:g}]")
    return value


def _color(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _COLOR.match(value):
        raise SceneError(f"{name} must be a #hex colour or a plain colour name")
    return value


def validate_scene(scene: Any) -> dict[str, Any]:
    """Return a sanitised copy of ``scene`` or raise SceneError."""
    if not isinstance(scene, dict):
        raise SceneError("render() must return a dict")
    width = _number(scene.get("width"), "scene.width", 1, 4000)
    height = _number(scene.get("height"), "scene.height", 1, 4000)
    items_in = scene.get("items", [])
    if not isinstance(items_in, (list, tuple)):
        raise SceneError("scene.items must be a list")
    if len(items_in) > MAX_ITEMS:
        raise SceneError(f"scene has more than {MAX_ITEMS} items")
    out_items: list[dict[str, Any]] = []
    for index, item in enumerate(items_in):
        label = f"items[{index}]"
        if not isinstance(item, dict):
            raise SceneError(f"{label} must be a dict")
        kind = item.get("type")
        if kind not in _ITEM_FIELDS:
            raise SceneError(f"{label}.type must be one of {sorted(_ITEM_FIELDS)}")
        clean: dict[str, Any] = {"type": kind}
        for name in _ITEM_FIELDS[kind]:
            if name not in item:
                raise SceneError(f"{label} ({kind}) is missing '{name}'")
            clean[name] = _number(item[name], f"{label}.{name}")
        if kind == "polygon":
            points = item.get("points")
            if not isinstance(points, (list, tuple)) or not 3 <= len(points) <= 256:
                raise SceneError(f"{label}.points must list 3 to 256 [x, y] points")
            clean["points"] = []
            for p_index, point in enumerate(points):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise SceneError(f"{label}.points[{p_index}] must be [x, y]")
                clean["points"].append(
                    [_number(point[0], f"{label}.points[{p_index}][0]"),
                     _number(point[1], f"{label}.points[{p_index}][1]")]
                )
        if kind == "text":
            text = item.get("text")
            if not isinstance(text, str) or len(text) > MAX_TEXT:
                raise SceneError(f"{label}.text must be a string of at most {MAX_TEXT} characters")
            clean["text"] = text
            anchor = item.get("anchor", "middle")
            if anchor not in _ANCHORS:
                raise SceneError(f"{label}.anchor must be one of {sorted(_ANCHORS)}")
            clean["anchor"] = anchor
        for name in _OPTIONAL_NUMBERS:
            if name in item and item[name] is not None:
                clean[name] = _number(item[name], f"{label}.{name}")
        if "opacity" in clean and not 0.0 <= clean["opacity"] <= 1.0:
            raise SceneError(f"{label}.opacity must be within [0, 1]")
        for name in ("fill", "stroke"):
            if item.get(name) is not None:
                clean[name] = _color(item[name], f"{label}.{name}")
        effect = item.get("effect", "none")
        if effect not in _EFFECTS:
            raise SceneError(f"{label}.effect must be one of {sorted(_EFFECTS)}")
        if effect != "none":
            clean["effect"] = effect
        if item.get("action") is not None:
            action = item["action"]
            if isinstance(action, bool) or not isinstance(action, int) or action < 0:
                raise SceneError(f"{label}.action must be a non-negative integer")
            clean["action"] = action
        if item.get("id") is not None:
            if not isinstance(item["id"], (str, int)) or len(str(item["id"])) > 64:
                raise SceneError(f"{label}.id must be a short string or integer")
            clean["id"] = str(item["id"])
        out_items.append(clean)
    result: dict[str, Any] = {"width": width, "height": height, "items": out_items}
    if scene.get("background") is not None:
        result["background"] = _color(scene["background"], "scene.background")
    hud = scene.get("hud")
    if hud is not None:
        if not isinstance(hud, str) or len(hud) > 400:
            raise SceneError("scene.hud must be a string of at most 400 characters")
        result["hud"] = hud
    return result
