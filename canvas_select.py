"""Bidirectional rectangle selector for a rendered PDF page."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "paper_audio_rectangle_selector",
    path=Path(__file__).parent / "crosshair_component",
)


def render_crosshair_canvas_selector(
    png_bytes: bytes,
    width: int = 620,
    key: str = "pdf_canvas",
    initial: dict | None = None,
) -> dict | None:
    """Render the PDF image and return normalized rectangle coordinates.

    The returned dictionary contains ``x0``, ``y0``, ``x1`` and ``y1`` in the
    0-1 range. Unlike ``components.html``, this declared component participates
    in Streamlit's widget protocol and can send a value back to Python.
    """
    encoded = base64.b64encode(png_bytes).decode("ascii")
    value = _COMPONENT(
        image=f"data:image/png;base64,{encoded}",
        width=max(320, int(width)),
        initial=initial,
        default=initial,
        key=key,
    )
    if not isinstance(value, dict):
        return None
    try:
        coordinates = {name: float(value[name]) for name in ("x0", "y0", "x1", "y1")}
    except (KeyError, TypeError, ValueError):
        return None
    return {name: max(0.0, min(1.0, coordinate)) for name, coordinate in coordinates.items()}
