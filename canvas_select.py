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
    key: str = "pdf_canvas",
    initial: dict | None = None,
    blocks: list[dict] | None = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
    max_height: int = 720,
) -> dict | None:
    """Render the PDF image and return normalized rectangle coordinates.

    The returned dictionary contains ``x0``, ``y0``, ``x1`` and ``y1`` in the
    0-1 range. Unlike ``components.html``, this declared component participates
    in Streamlit's widget protocol and can send a value back to Python.

    ``blocks`` holds the normalized text-block boxes of the page, from
    :func:`parser.normalized_block_boxes`. The component uses them to preview
    which paragraphs the rectangle captures and to select one on a click, so it
    must apply the same rule as :func:`parser.select_blocks_in_region`.
    """
    encoded = base64.b64encode(png_bytes).decode("ascii")
    value = _COMPONENT(
        image=f"data:image/png;base64,{encoded}",
        initial=initial,
        blocks=blocks or [],
        page_width=float(page_width),
        page_height=float(page_height),
        max_height=int(max_height),
        default=initial,
        key=key,
    )
    if not isinstance(value, dict):
        return None
    if value.get("cleared"):
        # Streamlit falls back to ``default`` when a component sends null, so
        # the component signals an empty selection with an explicit payload.
        return None
    try:
        coordinates = {name: float(value[name]) for name in ("x0", "y0", "x1", "y1")}
    except (KeyError, TypeError, ValueError):
        return None
    return {name: max(0.0, min(1.0, coordinate)) for name, coordinate in coordinates.items()}
