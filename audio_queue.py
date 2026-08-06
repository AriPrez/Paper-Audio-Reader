"""Player that accepts audio parts as they are generated and chains them."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "paper_audio_queue_player",
    path=Path(__file__).parent / "audio_queue_component",
)


def render_audio_queue(
    job: str,
    new_parts: list[tuple[int, bytes]],
    total_parts: int | None,
    done: bool,
    key: str = "audio_queue",
) -> None:
    """Hand newly finished parts to the player without interrupting playback.

    Only parts the component has not received yet are passed in. The component
    keeps every part it has been given for the life of its iframe, which
    survives reruns, so re-sending them would put a few megabytes of base64 on
    the websocket every time anything else on the page reruns — the audio would
    arrive faster and the interface would get slower. A part carries its index
    and is added once, so a repeated send is harmless if it ever happens.

    ``job`` identifies the recording. When it changes the component drops what
    it holds, otherwise a new selection would play after the previous one.
    """
    _COMPONENT(
        job=job,
        parts=[
            {"index": index, "data": base64.b64encode(audio).decode("ascii")}
            for index, audio in new_parts
        ],
        total_parts=int(total_parts) if total_parts else None,
        done=bool(done),
        default=None,
        key=key,
    )
